"""
agent/nodes/web_search_node.py

Searches the web using Tavily and builds a plan directly.
Does NOT go back through answer_planner — builds its own minimal plan
so response_renderer can format the result immediately.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from agent.state import AgentState, AnswerPlan

logger = logging.getLogger(__name__)


def web_search_node(state: AgentState) -> dict[str, Any]:
    logger.info("web_search_node: ENTERED")

    from config import settings
    from tavily import TavilyClient

    user_input      = state.get("user_input", "")
    collection_name = state.get("collection_name", "")
    analysis     = state.get("analysis", {}) or {}
    session      = state.get("session")

    # Extract car model from collection name for richer query
    # e.g. "biac_x55_ii_user_manual_en_5nyo" → "BIAC X55"
    SKIP_PARTS = {"ii", "iii", "iv", "user", "manual", "en"}
    car_parts  = collection_name.split("_") if collection_name else []
    clean_parts = []
    for p in car_parts:
        if p.lower() in SKIP_PARTS:
            continue
        if len(p) <= 4 and any(c.isdigit() for c in p):
            break
        clean_parts.append(p.upper())
    car_model = " ".join(clean_parts).strip() or "car"

    inferred_topic = analysis.get("inferred_topic", "")

    # Pull session context for additional grounding
    session_context = ""
    if session and hasattr(session, "to_context_string"):
        session_context = session.to_context_string()
    elif isinstance(session, dict) and session.get("issue_summary"):
        session_context = session.get("issue_summary", "")

    if inferred_topic and inferred_topic.lower() not in user_input.lower():
        query = f"{car_model} {inferred_topic}"
    else:
        query = f"{car_model} {user_input}"

    logger.info(f"web_search_node: inferred_topic='{inferred_topic}' query='{query}'")
    logger.info(f"web_search_node: extracted car_model='{car_model}' from collection='{collection_name}'")

    query = f"{car_model} {user_input}"
    logger.info(f"web_search_node: query='{query}'")

    try:
        client  = TavilyClient(api_key=settings.TAVILY_API_KEY)
        results = client.search(
            query,
            max_results=5,
            search_depth="advanced"
        )

        docs = []
        citations = []
        for r in results.get("results", []):
            content = r.get("content", "").strip()
            url     = r.get("url", "")
            title   = r.get("title", "")
            if not content:
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "source":   url,
                    "title":    title,
                    "filename": url,
                }
            ))
            citations.append({"page": "web", "section": title, "url": url})

        logger.info(f"web_search_node: got {len(docs)} results")

        # Build a plan directly — no need to call answer_planner again
        plan = AnswerPlan(
            mode="direct",
            confidence=0.85 if docs else 0.0,
            likely_goal=user_input,
            steps=None,
            expected_outcomes=None,
            safety_notes=[],
            citations=citations,
            first_clarifying_question=None,
            escalation_message=None if docs else (
                "Couldn't find relevant results online either. "
                "Try checking automotive review sites like CarAdvice or TopGear directly."
            ),
        )

        return {
            "source_nodes":         docs,
            "retrieval_successful": bool(docs),
            "search_used":          True,
            "plan":                 plan,
            "raw_answer":           "",
        }

    except Exception as e:
        logger.error(f"web_search_node failed: {e}")

        plan = AnswerPlan(
            mode="escalate",
            confidence=0.0,
            likely_goal=user_input,
            steps=None,
            expected_outcomes=None,
            safety_notes=[],
            citations=[],
            first_clarifying_question=None,
            escalation_message=(
                "Web search isn't available right now. "
                "Try checking automotive review sites directly for this comparison."
            ),
        )

        return {
            "source_nodes":         [],
            "retrieval_successful": False,
            "search_used":          True,
            "plan":                 plan,
            "raw_answer":           "",
        }