"""
The single TypedDict

Flow through the graph:
  User message
    → query_understanding  (fills: intent, expanded_queries, answer_mode)
    → memory_node          (fills: session, injects context into query)
    → retriever_node       (fills: raw_answer, source_nodes, confidence)
    → answer_planner       (fills: plan)
    → response_renderer    (fills: final_response)
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# ─── Session memory (persisted inside AgentState)
class SessionMemory(TypedDict, total = False):
    """
    What we know about this user's support session so far.
    Carried forward across every turn of the conversation.
    """
    product_model: str | None      
    issue_summary: str | None       
    attempted_steps: list[str]  
    current_stage: str                 # "initial" | "diagnosing" | "resolving" | "escalated"
    turn_count: int
    preferred_detail_level: str        # "brief" | "normal" | "detailed"
    last_intent: str | None            # carry forward for context

# ─── Query analysis result
class QueryAnalysis(TypedDict, total=False):
    """
    Output of query_understanding node.
    Describes what the user actually wants before we touch the retriever.
    """
    intent: str             # "faq" | "troubleshooting" | "how_to" | "page_request" | "comparison" | "followup"
    specificity: str        # "short" | "medium" | "detailed"
    answer_mode: str        # "direct" | "guided" | "troubleshoot" | "clarify"
    expanded_queries: list[str]       # 2-4 search variants for short queries
    needs_clarification: bool
    clarification_question: str | None
    inferred_topic: str | None  # what the system thinks the user means

# ─── Answer plan
class AnswerPlan(TypedDict, total=False):
    """
    Output of answer_planner node.
    Structured plan derived AFTER retrieval.
    response_renderer turns this into human prose — not free-form LLM output.
    """
    mode: str                          # "direct" | "step_by_step" | "troubleshoot" | "clarify" | "escalate"
    confidence: float                  # 0.0–1.0 — drives conditional routing
    likely_goal: str                   # "restore wifi connection"
    steps: list[str] | None            # ordered steps if mode is step_by_step/troubleshoot
    expected_outcomes: list[str] | None  # what user should see after each step
    safety_notes: list[str]            # "do not factory reset unless..."
    citations: list[dict]              # [{"page": 12, "section": "Wireless setup"}]
    first_clarifying_question: str | None  # only set when mode == "clarify"
    escalation_message: str | None     # only set when mode == "escalate"
 

# ─── Core AgentState
class AgentState(TypedDict, total=False):
    """
    The complete state object for one conversation thread.
 
    LangGraph passes this dict through every node in the graph.
    Each node receives the full state and returns only the keys it updates.
    The checkpointer serialises this to persistent storage per thread_id.
 
    Key design decisions:
    - messages uses add_messages reducer so history appends, never overwrites
    - All other keys use last-write-wins (LangGraph default)
    - session is a nested dict — updated by memory_node each turn
    """
    # ── Conversation history (LangGraph managed)
    messages: Annotated[list[Any], add_messages] # add_messages reducer
 
    # ── Current turn input
    user_input: str                    # raw user message this turn
    collection_name: str               # which PDF collection to query
    session_id: str                    # thread identifier for checkpointer
 
    # ── Query understanding output
    analysis: QueryAnalysis
    effective_query: str               # final query sent to retriever (may be expanded)
 
    # ── Session memory
    session: SessionMemory
 
    # ── Retrieval output
    raw_answer: str                    # LlamaIndex answer (unformatted)
    source_nodes: list[Any]            # LlamaIndex source nodes
    retrieval_successful: bool
 
    # ── Planning output
    plan: AnswerPlan
 
    # ── Final output
    final_response: str                # human prose — sent to user
    response_ready: bool               # signals end of graph

