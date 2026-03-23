"""
agent/nodes/response_renderer.py

LangGraph Node 5: Response Renderer

Input:  AgentState.plan, AgentState.analysis, AgentState.session,
        AgentState.user_input, AgentState.raw_answer
Output: AgentState.final_response, AgentState.response_ready

This node selects the correct prompt template based on plan.mode
and generates the final human prose response.

Why template-driven (not free-form)?
  - Predictable structure: troubleshoot always uses "Let's", "First", "Next"
  - Testable: you can assert response starts with expected pattern
  - Tunable: adjust a template without rewriting the node logic
  - The raw_answer from LlamaIndex already has page citations — we keep them

Streaming note:
  This node returns a complete string (non-streaming).
  Streaming is handled at the API layer (api/chat_router.py) separately,
  using a streaming-aware version of this node.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import AzureChatOpenAI

from agent.state import AgentState, AnswerPlan
from agent.prompts.system_prompt import RENDERER_PROMPTS, RENDERER_USER

logger = logging.getLogger(__name__)


def _get_llm(mode: str) -> AzureChatOpenAI:
    from config import settings
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_GPT4O_MINI_DEPLOYMENT,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=0.4,    # slight warmth — this is the conversational layer
        max_tokens=1200 if mode == "web_search" else 800,
    )


def response_renderer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function.
    """
    plan        = state.get("plan", {})
    user_input  = state.get("user_input", "")
    session     = state.get("session")
    analysis    = state.get("analysis", {})

    mode = plan.get("mode", "direct") if plan else "direct"

    # ── Select system prompt based on mode ────────────────────────────────
    system_prompt = RENDERER_PROMPTS.get(mode, RENDERER_PROMPTS["direct"])

    # ── Build session context ──────────────────────────────────────────────
    session_context = ""
    if session:
        if hasattr(session, 'to_context_string'):
            session_context = session.to_context_string()
        elif isinstance(session, dict) and session:
            session_context = str(session)

    # ── Serialise plan for prompt ──────────────────────────────────────────
    plan_json = json.dumps(plan, indent=2, default=str) if plan else "{}"

    raw_answer  = state.get("raw_answer", "")
    search_used = state.get("search_used", False)
    web_context = ""
    
    if raw_answer and search_used:
        web_context = f"Web search content — use this as your primary source:\n\n{raw_answer}"

    user_prompt = RENDERER_USER.format(
        session_context=session_context,
        user_input=user_input,
        plan_json=plan_json,
        web_context=web_context,
    )

    # ── Call LLM ──────────────────────────────────────────────────────────
    try:
        llm = _get_llm(mode=mode)
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ])

        final_response = response.content.strip()

        logger.info(
            f"response_renderer: mode={mode} | "
            f"length={len(final_response)} chars"
        )

        return {
            "final_response": final_response,
            "response_ready": True,
        }

    except Exception as e:
        logger.error(f"response_renderer: LLM call failed: {e}")

        # Graceful fallback: return raw_answer with an apology prefix
        # Better to give the user the retrieved answer than nothing
        raw = state.get("raw_answer", "")
        if not raw:
            docs = state.get("source_nodes", [])
            raw = "\n\n".join(
                f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in docs[:3]  # top 3 chunks is enough for a fallback
            )

        fallback = (
            f"{raw}\n\n(Note: I had trouble formatting this response. "
            f"The information above is from the manual.)"
        ) if raw else "I'm having trouble generating a response right now. Please try again."
        return {
            "final_response": fallback,
            "response_ready": True,
        }


# ─── Streaming variant ────────────────────────────────────────────────────────

def response_renderer_stream(state: AgentState):
    """
    Generator variant of response_renderer for WebSocket streaming.
    Used by api/chat_router.py — NOT called by the LangGraph graph directly.

    Yields tokens as they arrive from the LLM.
    Caller is responsible for collecting tokens and updating state.
    """
    plan       = state.get("plan", {})
    user_input = state.get("user_input", "")
    session    = state.get("session")

    mode = plan.get("mode", "direct") if plan else "direct"
    system_prompt = RENDERER_PROMPTS.get(mode, RENDERER_PROMPTS["direct"])

    session_context = ""
    if session:
        if hasattr(session, 'to_context_string'):
            session_context = session.to_context_string()

    plan_json = json.dumps(plan, indent=2, default=str) if plan else "{}"
    raw_answer  = state.get("raw_answer", "")
    search_used = state.get("search_used", False)
    web_context = ""
    if raw_answer and search_used:
        web_context = f"Web search content — use this as your primary source:\n\n{raw_answer}"

    user_prompt = RENDERER_USER.format(
        session_context=session_context,
        user_input=user_input,
        plan_json=plan_json,
        web_context=web_context,
    )

    try:
        # Build message list with full conversation history.
        # This is what makes "what's next?" work — the LLM sees the
        # previous exchange and continues naturally from it.
        messages_history = state.get("messages", [])
        prior_turns = messages_history[:-1]  # exclude current user turn

        llm_messages = [{"role": "system", "content": system_prompt}]
        for m in prior_turns[-6:]:  # last 3 exchanges = 6 messages
            if m.get("role") in ("user", "assistant") and m.get("content"):
                llm_messages.append({"role": m["role"], "content": m["content"]})
        llm_messages.append({"role": "user", "content": user_prompt})

        llm = _get_llm(mode=mode)
        for chunk in llm.stream(llm_messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"response_renderer_stream failed: {e}")
        yield f"\n[Error generating response: {e}]"