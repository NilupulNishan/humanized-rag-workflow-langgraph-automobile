"""
LangGraph State Machine — wires all nodes together.

Graph flow:
                    ┌─────────────────┐
                    │  query_understanding │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   memory_read    │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │    needs_clarification?     │ ◄── conditional edge
              └──────┬──────────────┬───────┘
                     │ NO           │ YES
          ┌──────────▼────┐   ┌─────▼──────────┐
          │  retriever    │   │  skip retrieval │
          └──────┬────────┘   │  (plan=clarify) │
                 │             └─────┬──────────┘
          ┌──────▼─────────┐         │
          │  answer_planner │         │
          └──────┬──────────┘         │
                 │                    │
              ┌──▼────────────────────▼──┐
              │     response_renderer     │
              └──────────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │   memory_write   │
                    └────────┬────────┘
                             │
                           END

Conditional routing after query_understanding:
  - needs_clarification=True  → skip retrieval, go direct to renderer
  - needs_clarification=False → full pipeline
 
Conditional routing after answer_planner:
  - mode == "escalate" → can add escalation handler here
  - all others → response_renderer
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

logger = logging.getLogger(__name__)

# ---- Conditional edge functions
def route_after_understanding(state: AgentState) -> str:
    """
    After query_understanding: should we skip retrieval?
    If the query needs clarification, we already know the plan.
    Go straight to a lightweight clarify plan, then render.
    """
    analysis = state.get("analysis", {})
    if analysis and analysis.get("needs_clarification", False):
        logger.debug("route: needs_clarification", False)
        return "skip_retrieval"
    return "memory_read"


def route_after_planner(state: AgentState) -> str:
    """
    After answer_planner: all modes go to response_renderer.
    This is a placeholder for future routing (e.g. human escalation).
    """
    plan = state.get("plan", {})
    mode = plan.get("mode", "direct") if plan else "direct"

    # Future: add escalation handler
    # if mode == "escalate":
    #     return "escalation_handler"
 
    return "response_renderer"

# ---- Skip-retrieval shortcut
def skip_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Shortcut node when clarification is needed before retrieval.
    Builds a minimal clarify plan directly from the analysis,
    bypassing the retriever and planner.
    """
    analysis = state.get("analysis", {})
    question = analysis.get("clarification_question") if analysis else None

    if not question:
        question = (
            "Could you give me a bit more details?"
            "That'll help me find the righ information for you"
        )
    
    from agent.state import AnswerPlan
    plan = AnswerPlan(
        mode="clarify",
        confidence=0.0,
        likely_goal=analysis.get("inferred_topic", "") if analysis else "",
        steps=None,
        expected_outcomes=None,
        safety_notes=[],
        citations=[],
        first_clarifying_question=question,
        escalation_message=None,
    )

    return {
        "plan": plan,
        "raw_answer": "",
        "source_nodes": [],
        "retrieval_successfull": False,
    }

# ---- Graph builder
def build_graph(use_persistence: bool = True) -> StateGraph:
    """
    Builds and compiles the LangGraph state machine.
 
    Args:
        use_persistence: If True, attaches MemorySaver checkpointer for
                         cross-turn state persistence. Set False for testing.
 
    Returns:
        Compiled graph ready for .invoke() or .stream()
    """
    graph = StateGraph(AgentState)
 
    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("memory_read",         memory_read_node)
    graph.add_node("skip_retrieval",      skip_retrieval_node)
    graph.add_node("retriever",           retriever_node)
    graph.add_node("answer_planner",      answer_planner_node)
    graph.add_node("response_renderer",   response_renderer_node)
    graph.add_node("memory_write",        memory_write_node)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("query_understanding")

    # ── Edges ─────────────────────────────────────────────────────────────
 
    # After understanding: route based on clarification flag
    graph.add_conditional_edges(
        "query_understanding",
        route_after_understanding,
        {
            "memory_read": "memory_read",
            "skip_retrieval": "skip_retrieval",
        }
    )
    # Normal path: memory_read → retriever → planner → renderer
    graph.add_edge("memory_read",    "retriever")
    graph.add_edge("retriever",      "answer_planner")
    # After planner: route (currently all → renderer)
    graph.add_conditional_edges(
        "answer_planner",
        route_after_planner,
        {"response_renderer": "response_renderer"}
    )
    # Clarify shortcut: skip_retrieval → renderer
    graph.add_edge("skip_retrieval", "response_renderer")

    # After render: save memory then end
    graph.add_edge("response_renderer", "memory_write")
    graph.add_edge("memory_write", END)


    # ── Compile ───────────────────────────────────────────────────────────
    if use_persistence:
        checkpointer = MemorySaver()
        compiled = graph.compile(checkpointer=checkpointer)
        logger.info("Graph compiled with MemorySaver checkpointer")
    else:
        compiled = graph.compile()
        logger.info("Graph compiled without persistence (test mode)")
 
    return compiled

# ---- Singleton
_graph = None

def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph(use_persistence=True)
    return _graph

# ---- High-level invoke 
 
def chat(
    user_input: str,
    session_id: str,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper for the graph.
 
    Args:
        user_input:      The user's message
        session_id:      Unique thread ID — LangGraph uses this for checkpointing
        collection_name: Which LlamaIndex collection to query (None = auto)
 
    Returns:
        Dict with at minimum: final_response, plan, source_nodes, session
    """
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
        f"mode={result.get('plan', {}).get('mode', '?')} | "
        f"response_len={len(result.get('final_response', ''))}"
    )
 
    return result