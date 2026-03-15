"""
LangGraph Node 1: Query Understanding
 
Input:  AgentState.user_input, AgentState.session
Output: AgentState.analysis, AgentState.effective_query
 
What it does:
  1. Reads user's raw message + session context
  2. Calls LLM (cheap, fast — no retrieval yet) to classify intent
  3. If query is short/vague: expands into 2-4 search variants
  4. If query needs clarification: flags it (graph will route to clarify node)
  5. Builds effective_query — the actual string(s) sent to the retriever
 
This node is the "human support engineer's first read" of the message.
It decides HOW to approach answering before touching the vector DB.
"""
from __future__ import annotations
 
import json
import logging
from typing import Any
 
from langchain_openai import AzureChatOpenAI

from agent.state import AgentState, QueryAnalysis
from agent.prompts.system_prompt import (
    QUERY_UNDERSTANDING_SYSTEM,
    QUERY_UNDERSTANDING_USER,
)

logger = logging.getLogger(__name__)

def _get_llm() -> AzureChatOpenAI:
    from config import settings
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_GPT4O_MINI_DEPLOYMENT,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=0,           # deterministic classification
        max_tokens=400,          # analysis is small
    )

def query_understanding_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function.
    Must return a dict of keys to update in AgentState.
    """
    user_input = state.get("user_input", "").strip()
    session = state.get("session", {})

    if not user_input:
        logger.warning("query_understanding: empty user_input")
        return {
            "analysis": QueryAnalysis(
                intent="faq",
                specificity="short",
                answer_mode="direct",
                expanded_queries=[],
                needs_clarification=False,
                clarification_question=None,
                inferred_topic="unknown",
            ),
            "effective_query": user_input,
        }
    
    # Build session context string for injection into prompt
    session_context = ""
    if hasattr(session, 'to_context_string'):
        session_context = session.to_context_string()
    elif isinstance(session, dict) and session:
        parts = []
        if session.get("issue_summary"):
            parts.append(f"Issue: {session['issue_summary']}")
        if session.get("attempted_steps"):
            parts.append(f"Already tried: {', '.join(session['attempted_steps'])}")
        if parts:
            session_context = "[Session: " + " | ".join(parts) + "]"

    # Build prompt
    user_prompt = QUERY_UNDERSTANDING_USER.format(
        session_context=session_context,
        user_input=user_input,
    )

    # Call LLM
    try:
        llm = _get_llm()
        response = llm.invoke([
            {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ])

        raw = response.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        raw = raw.strip()
        data = json.loads(raw)

        analysis = QueryAnalysis(
            intent = data.get("intent", "faq"),
            specificity = data.get("specificity", "medium"),
            answer_mode = data.get("answer_mode", "direct"),
            expanded_queries = data.get("expanded_queries", []),
            needs_clarification = data.get("needs_clarification", False),
            clarification_question = data.get("clarification_question"),
            inferred_topic = data.get("inferred_topic", user_input),
        )

        # Determine effective query for retriever
        # - Short queries: use expanded set (retriever_node will handle multi-query)
        # - Otherwise: inject session context for richer retrieval
        if analysis["specificity"] == "short" and analysis["expanded_queries"]:
            effective_query = user_input  # retriever_node will use expanded_queries
        elif session_context:
            effective_query = f"{session_context}\n{user_input}"
        else:
            effective_query = user_input
 
        logger.info(
            f"query_understanding: intent={analysis['intent']} "
            f"mode={analysis['answer_mode']} "
            f"specificity={analysis['specificity']} "
            f"expanded={len(analysis.get('expanded_queries', []))} variants"
        )

        return {
            "analysis": analysis,
            "effective_query": effective_query,
        }
 
    except json.JSONDecodeError as e:
        logger.error(f"query_understanding: JSON parse failed: {e}")
        # Graceful degradation — treat as a direct medium-specificity query
        return {
            "analysis": QueryAnalysis(
                intent="faq",
                specificity="medium",
                answer_mode="direct",
                expanded_queries=[],
                needs_clarification=False,
                clarification_question=None,
                inferred_topic=user_input,
            ),
            "effective_query": user_input,
        }
    
    except Exception as e:
        logger.error(f"query_understanding: LLM call failed: {e}")
        return {
            "analysis": QueryAnalysis(
                intent="faq",
                specificity="medium",
                answer_mode="direct",
                expanded_queries=[],
                needs_clarification=False,
                clarification_question=None,
                inferred_topic=user_input,
            ),
            "effective_query": user_input,
        }