"""
LangGraph Node 2: Memory Read/Write
 
Input:  AgentState.session_id, AgentState.user_input, AgentState.analysis
Output: AgentState.session (updated)
 
Two responsibilities:
  READ  — load existing session before retrieval so context enriches the query
  WRITE — update session after response with new facts extracted from this turn
 
The graph calls this node TWICE:
  1. Before retriever_node (READ pass) — load session
  2. After response_renderer (WRITE pass) — extract new facts, save
 
A single node handles both passes via the `mode` parameter in state.
This avoids duplicating the store access logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.memory.schemas import SessionData
from agent.memory.session_store import get_session_store
from agent.state import AgentState


logger = logging.getLogger(__name__)

def memory_read_node(state: AgentState) -> dict[str, Any]:
    """
    READ pass — called before retrieval.
    Loads existing session from store and returns it into AgentState.
    """
    session_id = state.get("session_id", "default")
    store = get_session_store()
    session = store.get_or_create(session_id)

    # Increment turn counter
    session.turn_count += 1

    #  Carry forward the current intent if analysis is available
    analysis = state.get("analysis", {})
    if analysis and analysis.get("intent"):
        session.last_intent = analysis["intent"]

    # Set collection from state if session doesn't have it yet
    if not session.collection_name:
        collection = state.get("collection_name")
        if collection:
            session.collection_name = collection

    store.save(session_id, session)
    logger.debug(f"memory_read: session {session_id} | turn {session.turn_count}")

    return {"session": session}

def memory_write_node(state: AgentState) -> dict[str, Any]:
    session_id = state.get("session_id", "default")
    store      = get_session_store()
    session    = store.get(session_id)

    if not session:
        logger.warning(f"memory_write: session {session_id} not found, skipping")
        return {}

    user_input = state.get("user_input", "")
    analysis   = state.get("analysis", {}) or {}
    plan       = state.get("plan", {}) or {}

    # ── Product model ─────────────────────────────────────────────────────
    if not session.product_model:
        model = _extract_model(user_input)
        if model:
            session.product_model = model

    # ── BUG FIX 1: Always update issue_summary when topic changes ─────────
    # Old code: if not session.issue_summary → froze on first topic forever
    # New code: update whenever inferred_topic differs from current summary
    new_topic = analysis.get("inferred_topic", "")
    intent    = analysis.get("intent", "")

    if new_topic and new_topic.lower() not in ("unknown", "social/conversational"):
        # BUG FIX 2: Reset attempted_steps when topic changes significantly
        # Detect topic change: new topic doesn't overlap with current summary
        current = (session.issue_summary or "").lower()
        incoming = new_topic.lower()
        words_overlap = any(
            w in current for w in incoming.split()
            if len(w) > 4  # ignore short words like "the", "how"
        )
        if not words_overlap and session.issue_summary:
            # Topic has changed — reset context so old steps don't bleed in
            logger.info(
                f"memory_write: topic changed "
                f"'{session.issue_summary}' → '{new_topic}' — resetting steps"
            )
            session.attempted_steps = []
            session.current_stage   = "initial"

        session.issue_summary = new_topic

    # ── Stage advancement ─────────────────────────────────────────────────
    plan_mode = plan.get("mode", "")
    if plan_mode in ("troubleshoot", "step_by_step") and session.current_stage == "initial":
        session.advance_stage("diagnosing")
    elif plan_mode == "escalate":
        session.advance_stage("escalated")

    # ── Tried steps from user message ────────────────────────────────────
    tried = _extract_tried_steps(user_input)
    for step in tried:
        session.mark_step_attempted(step)

    # ── BUG FIX 3: use timezone-aware datetime ────────────────────────────
    from datetime import datetime, timezone
    session.last_active = datetime.now(timezone.utc)

    store.save(session_id, session)
    logger.debug(
        f"memory_write: {session_id} | "
        f"topic='{session.issue_summary}' | "
        f"stage={session.current_stage} | "
        f"tried={session.attempted_steps}"
    )

    return {"session": session}

# helpers ------------------------------------------
def _extract_model(text: str) -> str | None:
    """
    Extract product model numbers from user text.
    Heuristic patterns — extend for your specific product lines.
    """
    known_models = [
        "HALO24",
        "R3016",
        "QUINT-PS/1AC/24DC/5",
        "1085173",
        "OEM",
        "CL5708",
        "120TV",
        "2153449-4",
        "32HFL3014",
        "H3114AV",
        "H3118AV",
        "R740",
        "TMP5x24",
        "C1300-24FP-4X",
        "S5735-L24T4S-A-V",
        "CLEAR H98MUV3",
        "LMXP"
    ]
    # Detect known models first
    for model in known_models:
        if re.search(rf'\b{re.escape(model)}\b', text, re.IGNORECASE):
            return model
        
    # Generic patterns
    patterns = [
        r'\b[A-Z]{1,4}[\s\-]?[A-Z0-9]{2,8}\b',     # HP M428, X200
        r'\bmodel\s+([A-Z0-9\-\/]+)\b',            # model X200
        r'\b[A-Z][a-z]+[A-Z][a-z]+\s+\w+\s+\d+',   # LaserJet Pro 4104
        r'\b[A-Z0-9]+(?:[-/][A-Z0-9]+)+\b',        # C1300-24FP-4X, QUINT-PS/1AC/24DC/5
        r'\b[A-Z]{1,5}\d{2,6}[A-Z]*\b'             # R740, HALO24, H3114AV
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return None


def _extract_tried_steps(text: str) -> list[str]:
    """
    Extract what user says they've already tried.

    patterns:
        "I already tried restarting"
        "I restarted the printer"
        "tried turning it off"
        "already checked the cable"
        "I also did restart"
        "I've done restart", "I have done reset"
        "I attempted restart"
        "I gave restarting a try"
        "I tried restarting it already"
        NEGATIVE patterns (to capture what user DIDN'T try)
    """

    tried = []
    patterns = [
    r"(?:already\s+)?tried\s+([\w\s]+?)(?:\s+and|\s+but|\.|,|$)",
    r"(?:already\s+)?(?:restarted|rebooted|reset|checked|unplugged|reconnected)\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:have\s+)?(?:already\s+)?(?:tried|done|checked)\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:already\s+)?did\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:also\s+)?did\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:have\s+|'ve\s+)?done\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:have\s+)?attempted\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:gave|give)\s+([\w\s]+?)\s+(?:a\s+)?try",
    r"(?:already\s+)?tried\s+([\w\s]+?)\s+(?:already)?(?:\.|,|$)",
    # NEGATIVE patterns (to capture what user DIDN'T try)
    r"(?:did\s+not|didn't)\s+try\s+([\w\s]+?)(?:\.|,|$)",
    r"(?:haven't|have\s+not)\s+tried\s+([\w\s]+?)(?:\.|,|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            step = m.strip()
            if len(step) > 3:  # filter noise
                tried.append(step)
    return tried



    
