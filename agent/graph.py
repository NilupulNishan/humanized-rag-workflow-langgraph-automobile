"""
agent/graph.py

LangGraph State Machine
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes.query_understanding import query_understanding_node
from agent.nodes.memory_node import memory_read_node, memory_write_node
from agent.nodes.retriever_node import retriever_node
from agent.nodes.answer_planner import answer_planner_node
from agent.nodes.response_renderer import response_renderer_node
from agent.nodes.web_search_node import web_search_node

logger = logging.getLogger(__name__)


def direct_answer_node(state: AgentState) -> dict[str, Any]:
    from langchain_openai import AzureChatOpenAI
    from agent.state import AnswerPlan

    user_input       = state.get("user_input", "")
    messages_history = state.get("messages", [])

    GENERAL_SYSTEM = """\
You are VivoAssist, a friendly and knowledgeable technical support assistant.
The user has sent a message that doesn't require looking up the product manual.
Respond naturally and helpfully as a warm assistant would.
Keep responses concise. Don't mention the manual unless relevant.
Don't add filler phrases like "Great question!" or "Certainly!".
"""
    llm_messages = [{"role": "system", "content": GENERAL_SYSTEM}]
    for m in messages_history[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            llm_messages.append({"role": m["role"], "content": m["content"]})

    try:
        from config import settings
        llm = AzureChatOpenAI(
            azure_deployment=settings.AZURE_GPT4O_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            max_tokens=400,
        )
        response = llm.invoke(llm_messages)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"direct_answer_node failed: {e}")
        answer = "I'm here to help. Could you tell me more about what you need?"

    logger.info(f"direct_answer_node: answered general query '{user_input[:60]}'")

    plan = AnswerPlan(
        mode="direct", confidence=1.0, likely_goal="general conversation",
        steps=None, expected_outcomes=None, safety_notes=[], citations=[],
        first_clarifying_question=None, escalation_message=None,
    )

    return {
        "raw_answer":           answer,
        "source_nodes":         [],
        "retrieval_successful": True,
        "plan":                 plan,
        "final_response":       answer,
        "response_ready":       True,
    }


def skip_retrieval_node(state: AgentState) -> dict[str, Any]:
    from agent.state import AnswerPlan

    analysis = state.get("analysis", {})
    question = analysis.get("clarification_question") if analysis else None
    if not question:
        question = "Could you give me a bit more detail? That'll help me find the right information for you."

    plan = AnswerPlan(
        mode="clarify", confidence=0.0,
        likely_goal=analysis.get("inferred_topic", "") if analysis else "",
        steps=None, expected_outcomes=None, safety_notes=[], citations=[],
        first_clarifying_question=question, escalation_message=None,
    )

    return {
        "plan": plan, "raw_answer": "",
        "source_nodes": [], "retrieval_successful": False,
    }


# ─── Routing functions ────────────────────────────────────────────────────────

def route_after_understanding(state: AgentState) -> str:
    analysis = state.get("analysis") or {}
    intent   = analysis.get("intent", "")
    logger.info(f"route_after_understanding: intent='{intent}'")

    if intent == "general":
        return "direct_answer"

    if intent == "this_car_vs_another_comparison":
        logger.info("route: comparison → web_search (skipping retriever)")
        return "web_search"

    if analysis.get("needs_clarification", False):
        return "skip_retrieval"

    return "memory_read"


def route_after_planner(state: AgentState) -> str:
    plan        = state.get("plan", {})
    search_used = state.get("search_used", False)

    logger.info(f"route_after_planner: mode={plan.get('mode')} confidence={plan.get('confidence')} search_used={search_used}")

    # BUG FIX: already used web search — never loop back
    if search_used:
        logger.info("route_after_planner: search_used=True → response_renderer")
        return "response_renderer"

    mode       = (plan or {}).get("mode", "direct")
    confidence = (plan or {}).get("confidence", 1.0)

    if mode == "web_search_needed":
        return "web_search"

    if mode == "escalate" or confidence < 0.35:
        return "web_search"

    return "response_renderer"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph(use_persistence: bool = True) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("direct_answer",       direct_answer_node)
    graph.add_node("memory_read",         memory_read_node)
    graph.add_node("skip_retrieval",      skip_retrieval_node)
    graph.add_node("retriever",           retriever_node)
    graph.add_node("answer_planner",      answer_planner_node)
    graph.add_node("web_search",          web_search_node)
    graph.add_node("response_renderer",   response_renderer_node)
    graph.add_node("memory_write",        memory_write_node)

    graph.set_entry_point("query_understanding")

    graph.add_conditional_edges(
        "query_understanding",
        route_after_understanding,
        {
            "direct_answer":  "direct_answer",
            "skip_retrieval": "skip_retrieval",
            "web_search":     "web_search",
            "memory_read":    "memory_read",
        }
    )

    graph.add_edge("direct_answer", "memory_write")
    graph.add_edge("memory_read",   "retriever")
    graph.add_edge("retriever",     "answer_planner")

    graph.add_conditional_edges(
        "answer_planner",
        route_after_planner,
        {
            "web_search":        "web_search",
            "response_renderer": "response_renderer",
        }
    )

    # BUG FIX: web_search → response_renderer directly
    # web_search_node builds its own plan so answer_planner is not needed again
    graph.add_edge("web_search",     "response_renderer")
    graph.add_edge("skip_retrieval", "response_renderer")
    graph.add_edge("response_renderer", "memory_write")
    graph.add_edge("memory_write",       END)

    if use_persistence:
        checkpointer = MemorySaver()
        compiled = graph.compile(checkpointer=checkpointer)
        logger.info("Graph compiled with MemorySaver checkpointer")
    else:
        compiled = graph.compile()
        logger.info("Graph compiled without persistence (test mode)")

    return compiled


# ─── Singleton ────────────────────────────────────────────────────────────────

_graph = None


def get_graph(force_rebuild: bool = False):
    global _graph
    if _graph is None or force_rebuild:
        _graph = build_graph(use_persistence=True)
    return _graph


# ─── High-level invoke ────────────────────────────────────────────────────────

def chat(
    user_input: str,
    session_id: str,
    collection_name: str | None = None,
) -> dict[str, Any]:
    graph = get_graph()

    initial_state: AgentState = {
        "user_input":      user_input,
        "session_id":      session_id,
        "collection_name": collection_name or "",
        "messages":        [{"role": "user", "content": user_input}],
    }

    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(initial_state, config=config)

    logger.info(
        f"chat: session={session_id} | "
        f"intent={result.get('analysis', {}).get('intent', '?')} | "
        f"mode={result.get('plan', {}).get('mode', '?')} | "
        f"len={len(result.get('final_response', ''))}"
    )
    return result