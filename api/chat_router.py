from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    collection: Optional[str] = None


class SourceInfo(BaseModel):
    page: Optional[int | str] = None
    section: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    mode: str
    confidence: float
    sources: list[SourceInfo]
    needs_followup: bool


def _run_pipeline(
    user_input: str,
    session_id: str,
    collection_name: str | None,
) -> dict:
    """
    Shared pipeline logic for both REST and WebSocket endpoints.
    Mirrors the routing logic in app.py run_pipeline().
    """
    from agent.nodes.query_understanding import query_understanding_node
    from agent.nodes.memory_node import memory_read_node, memory_write_node
    from agent.nodes.retriever_node import retriever_node
    from agent.nodes.answer_planner import answer_planner_node
    from agent.nodes.web_search_node import web_search_node
    from agent.nodes.response_renderer import response_renderer_node
    from agent.state import AgentState, AnswerPlan

    state: AgentState = {
        "user_input":      user_input,
        "session_id":      session_id,
        "collection_name": collection_name or "",
        "messages":        [{"role": "user", "content": user_input}],
    }

    # ── Node 1: query understanding ───────────────────────────────────────
    result = query_understanding_node(state)
    result.pop("collection_name", None)
    state.update(result)

    analysis   = state.get("analysis", {}) or {}
    intent     = analysis.get("intent", "")
    needs_clarification = analysis.get("needs_clarification", False)

    # ── Route based on intent ─────────────────────────────────────────────
    if intent == "general":
        from agent.graph import direct_answer_node
        result = direct_answer_node(state)
        result.pop("collection_name", None)
        state.update(result)

    elif intent == "this_car_vs_another_comparison":
        logger.info("chat_router: comparison → web_search")
        result = web_search_node(state)
        result.pop("collection_name", None)
        state.update(result)

    elif needs_clarification:
        question = analysis.get("clarification_question",
            "Could you give me more detail? That'll help me find the right answer.")
        state["plan"] = AnswerPlan(
            mode="clarify", confidence=0.0,
            likely_goal=analysis.get("inferred_topic", ""),
            steps=None, expected_outcomes=None, safety_notes=[],
            citations=[], first_clarifying_question=question,
            escalation_message=None,
        )
        state["raw_answer"] = ""
        state["source_nodes"] = []
        state["retrieval_successful"] = False

    else:
        # Normal pipeline
        for fn in [memory_read_node, retriever_node, answer_planner_node]:
            result = fn(state)
            result.pop("collection_name", None)
            state.update(result)

        # Post-planner web search fallback
        plan        = state.get("plan", {}) or {}
        mode        = plan.get("mode", "direct")
        confidence  = plan.get("confidence", 1.0)
        search_used = state.get("search_used", False)

        if not search_used and (mode in ("web_search_needed", "escalate") or confidence < 0.35):
            result = web_search_node(state)
            result.pop("collection_name", None)
            state.update(result)

    # ── Renderer ──────────────────────────────────────────────────────────
    if not state.get("final_response"):
        result = response_renderer_node(state)
        result.pop("collection_name", None)
        state.update(result)

    # ── Memory write ──────────────────────────────────────────────────────
    memory_write_node(state)

    return state


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    import asyncio

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_pipeline(
                user_input=request.message,
                session_id=session_id,
                collection_name=request.collection,
            )
        )

        plan         = result.get("plan", {}) or {}
        source_nodes = result.get("source_nodes", []) or []

        sources = []
        for node in source_nodes:
            meta = getattr(node, "metadata", {}) or {}
            page = meta.get("page_number") or meta.get("page") or meta.get("page_label")
            section = meta.get("section") or meta.get("header") or ""
            if page:
                try:
                    sources.append(SourceInfo(page=int(page), section=section))
                except (ValueError, TypeError):
                    # Web results have no integer page — skip
                    pass

        return ChatResponse(
            session_id=session_id,
            response=result.get("final_response", ""),
            mode=plan.get("mode", "direct"),
            confidence=float(plan.get("confidence", 0.5)),
            sources=sources,
            needs_followup=plan.get("mode") in ("clarify", "troubleshoot"),
        )

    except Exception as e:
        logger.error(f"chat_endpoint error: {e}")
        return ChatResponse(
            session_id=session_id,
            response="I encountered an error processing your request. Please try again.",
            mode="error",
            confidence=0.0,
            sources=[],
            needs_followup=False,
        )


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data       = json.loads(raw)
                user_input = data.get("message", "").strip()
                collection = data.get("collection")
            except json.JSONDecodeError:
                user_input = raw.strip()
                collection = None

            if not user_input:
                continue

            logger.info(f"WS [{session_id}]: '{user_input[:60]}'")

            try:
                await _handle_ws_message(
                    websocket=websocket,
                    user_input=user_input,
                    session_id=session_id,
                    collection_name=collection,
                )
            except Exception as e:
                logger.error(f"WS message handler error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": str(e)
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


async def _handle_ws_message(
    websocket: WebSocket,
    user_input: str,
    session_id: str,
    collection_name: str | None,
):
    import asyncio
    from agent.nodes.query_understanding import query_understanding_node
    from agent.nodes.memory_node import memory_read_node, memory_write_node
    from agent.nodes.retriever_node import retriever_node
    from agent.nodes.answer_planner import answer_planner_node
    from agent.nodes.web_search_node import web_search_node
    from agent.nodes.response_renderer import response_renderer_stream
    from agent.state import AgentState, AnswerPlan

    state: AgentState = {
        "user_input":      user_input,
        "session_id":      session_id,
        "collection_name": collection_name or "",
        "messages":        [{"role": "user", "content": user_input}],
    }

    loop = asyncio.get_event_loop()

    def _run(fn):
        result = fn(state)
        result.pop("collection_name", None)
        state.update(result)

    # ── Node 1 ────────────────────────────────────────────────────────────
    await loop.run_in_executor(None, lambda: _run(query_understanding_node))

    analysis   = state.get("analysis", {}) or {}
    intent     = analysis.get("intent", "")
    needs_clarification = analysis.get("needs_clarification", False)

    # ── Route ─────────────────────────────────────────────────────────────
    if intent == "general":
        from agent.graph import direct_answer_node
        await loop.run_in_executor(None, lambda: _run(direct_answer_node))

    elif intent == "this_car_vs_another_comparison":
        await loop.run_in_executor(None, lambda: _run(web_search_node))

    elif needs_clarification:
        question = analysis.get("clarification_question",
            "Could you give me more detail?")
        state["plan"] = AnswerPlan(
            mode="clarify", confidence=0.0,
            likely_goal=analysis.get("inferred_topic", ""),
            steps=None, expected_outcomes=None, safety_notes=[],
            citations=[], first_clarifying_question=question,
            escalation_message=None,
        )
        state["raw_answer"] = ""
        state["source_nodes"] = []
        state["retrieval_successful"] = False

    else:
        await loop.run_in_executor(None, lambda: _run(memory_read_node))
        await loop.run_in_executor(None, lambda: _run(retriever_node))
        await loop.run_in_executor(None, lambda: _run(answer_planner_node))

        plan        = state.get("plan", {}) or {}
        mode        = plan.get("mode", "direct")
        confidence  = plan.get("confidence", 1.0)
        search_used = state.get("search_used", False)

        if not search_used and (mode in ("web_search_needed", "escalate") or confidence < 0.35):
            await loop.run_in_executor(None, lambda: _run(web_search_node))

    # ── Streaming renderer ────────────────────────────────────────────────
    full_response = ""

    def _generate_tokens():
        return list(response_renderer_stream(state))

    tokens = await loop.run_in_executor(None, _generate_tokens)

    for token in tokens:
        full_response += token
        await websocket.send_text(json.dumps({"type": "token", "data": token}))

    state["final_response"] = full_response
    state["response_ready"] = True

    await loop.run_in_executor(None, lambda: memory_write_node(state))

    # ── Send metadata ─────────────────────────────────────────────────────
    plan         = state.get("plan", {}) or {}
    source_nodes = state.get("source_nodes", []) or []
    search_used  = state.get("search_used", False)

    sources = []
    for node in source_nodes:
        meta = getattr(node, "metadata", {}) or {}
        if search_used:
            # Web results — send URL instead of page number
            url   = meta.get("source", "")
            title = meta.get("title", url)
            if url:
                sources.append({"page": "web", "section": title, "url": url})
        else:
            page    = meta.get("page_number") or meta.get("page") or meta.get("page_label")
            section = meta.get("section") or meta.get("header") or ""
            if page:
                try:
                    sources.append({"page": int(page), "section": section})
                except (ValueError, TypeError):
                    pass

    if sources:
        await websocket.send_text(json.dumps({"type": "sources", "data": sources}))

    await websocket.send_text(json.dumps({
        "type": "plan",
        "data": {
            "mode":       plan.get("mode", "direct"),
            "confidence": plan.get("confidence", 0.5),
            "likely_goal": plan.get("likely_goal", ""),
        }
    }))

    await websocket.send_text(json.dumps({"type": "done", "data": None}))


def slugify_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\.pdf$", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


@router.get("/collections")
async def list_collections():
    try:
        pdf_dir = Path("data/pdfs")
        items   = []
        for pdf_file in pdf_dir.glob("*.pdf"):
            collection_id = slugify_filename(pdf_file.name)
            items.append({
                "id":         collection_id,
                "title":      pdf_file.stem,
                "pdf_url":    f"/pdfs/{pdf_file.name}",
                "filename":   pdf_file.name,
            })
        return {"collections": items}
    except Exception as e:
        return {"collections": [], "error": str(e)}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        from agent.memory.session_store import get_session_store
        store = get_session_store()
        store.delete(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        return {"status": "error", "detail": str(e)}