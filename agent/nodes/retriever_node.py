"""
LangGraph Node 3: Retrieval

Input:  AgentState.effective_query, AgentState.analysis, AgentState.collection_name
Output: AgentState.raw_answer, AgentState.source_nodes, AgentState.retrieval_successful

This node is the ONLY place in the graph that touches the retriever.
All retrieval logic stays in src/retriever.py — untouched.
This node just calls it and maps results back into AgentState.

Two retrieval paths:
  1. Short/expanded query  → retrieve_expanded() runs all variants, deduplicates, merges
  2. Normal query          → single effective_query via langchain retriever.invoke()

NOTE: The new retriever (RetrieverManager) returns LangChain Document objects,
not a result wrapper. source_nodes here are those Document objects.
raw_answer is left empty — the LLM synthesis node is responsible for generating
the final answer from source_nodes (documents).
"""
from __future__ import annotations

import logging
from typing import Any

from agent.state import AgentState
from src.retriever import RetrieverManager

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
# RetrieverManager loads all ChromaDB collections once at startup.
# Re-used across every graph invocation — no re-loading on each call.

_retriever_manager: RetrieverManager | None = None


def _get_manager() -> RetrieverManager:
    global _retriever_manager
    if _retriever_manager is None:
        _retriever_manager = RetrieverManager()
        _retriever_manager.load_all()
    return _retriever_manager


def _retrieve_expanded(retriever, queries: list[str]) -> list:
    """
    Run multiple query variants and return a deduplicated list of Documents.

    Deduplication is by page_content so the LLM doesn't see the same chunk
    twice even if several query variants matched it.
    """
    seen: set[str] = set()
    merged: list = []

    for q in queries:
        try:
            docs = retriever.invoke(q)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
        except Exception as e:
            logger.warning(f"retriever_node: expanded variant failed '{q[:50]}': {e}")

    return merged


# ── Node ──────────────────────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function.
    """
    collection_name = state.get("collection_name")
    effective_query = state.get("effective_query", "")
    analysis        = state.get("analysis", {})

    if not effective_query:
        logger.error("retriever_node: effective_query is empty")
        return {
            "raw_answer":           "",
            "source_nodes":         [],
            "retrieval_successful": False,
        }

    # ── Resolve retriever ─────────────────────────────────────────────────────
    try:
        manager   = _get_manager()
        retriever = manager.get_retriever(collection_name) if collection_name \
                    else _get_any_retriever(manager)
    except (KeyError, ValueError) as e:
        logger.error(f"retriever_node: could not load retriever: {e}")
        return {
            "raw_answer":           "",
            "source_nodes":         [],
            "retrieval_successful": False,
        }

    # ── Choose retrieval strategy ─────────────────────────────────────────────
    expanded_queries = analysis.get("expanded_queries", []) if analysis else []
    specificity      = analysis.get("specificity", "medium") if analysis else "medium"

    try:
        if specificity == "short" and expanded_queries:
            logger.info(
                f"retriever_node: expanded retrieval | "
                f"{len(expanded_queries)} variants | collection={collection_name}"
            )
            docs = _retrieve_expanded(retriever, expanded_queries)

        else:
            logger.info(
                f"retriever_node: single retrieval | "
                f"query='{effective_query[:60]}...' | collection={collection_name}"
            )
            docs = retriever.invoke(effective_query)

    except Exception as e:
        logger.error(f"retriever_node: retrieval failed: {e}")
        return {
            "raw_answer":           "",
            "source_nodes":         [],
            "retrieval_successful": False,
        }

    if not docs:
        logger.warning("retriever_node: retrieval returned no documents")

    # raw_answer is intentionally empty — the synthesis node builds the answer
    # from source_nodes (Documents). If your graph expects raw_answer to be
    # pre-filled, pass the concatenated page_content instead:
    #   raw_answer = "\n\n".join(d.page_content for d in docs)
    return {
        "raw_answer":           "",
        "source_nodes":         docs,
        "retrieval_successful": bool(docs),
    }


def _get_any_retriever(manager: RetrieverManager):
    """
    Fallback when no collection_name is specified in state.
    Uses the first available collection (mirrors old MultiRAGTool behaviour).
    """
    names = manager.get_collection_names()
    if not names:
        raise ValueError("No collections available in RetrieverManager")
    logger.info(f"retriever_node: no collection specified, defaulting to '{names[0]}'")
    return manager.get_retriever(names[0])