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


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    import asyncio
    from agent.graph import chat

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chat(
                user_input=request.message,
                session_id=session_id,
                collection_name=request.collection,
            )
        )

        plan = result.get("plan", {}) or {}
        source_nodes = result.get("source_nodes", []) or []

        sources = []
        for node in source_nodes:
            meta = getattr(node, "metadata", {}) or {}
            page = meta.get("page_number") or meta.get("page") or meta.get("page_label")
            section = meta.get("section") or meta.get("header") or ""
            if page:
                sources.append(SourceInfo(page=page, section=section))

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
                data = json.loads(raw)
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
    from agent.state import AgentState
    from agent.nodes.response_renderer import response_renderer_stream
    from agent.nodes.query_understanding import query_understanding_node
    from agent.nodes.memory_node import memory_read_node, memory_write_node
    from agent.nodes.retriever_node import retriever_node
    from agent.nodes.answer_planner import answer_planner_node
    from agent.state import AnswerPlan

    initial_state: AgentState = {
        "user_input": user_input,
        "session_id": session_id,
        "collection_name": collection_name or "",
        "messages": [{"role": "user", "content": user_input}],
    }

    state = dict(initial_state)
    loop = asyncio.get_event_loop()

    state.update(await loop.run_in_executor(None, lambda: query_understanding_node(state)))
    state.update(await loop.run_in_executor(None, lambda: memory_read_node(state)))

    analysis = state.get("analysis", {})
    if analysis and analysis.get("needs_clarification"):
        question = analysis.get("clarification_question", "Could you give me more detail?")
        state["plan"] = AnswerPlan(
            mode="clarify",
            confidence=0.0,
            likely_goal=analysis.get("inferred_topic", ""),
            steps=None,
            expected_outcomes=None,
            safety_notes=[],
            citations=[],
            first_clarifying_question=question,
            escalation_message=None,
        )
        state["raw_answer"] = ""
        state["source_nodes"] = []
        state["retrieval_successful"] = False
    else:
        state.update(await loop.run_in_executor(None, lambda: retriever_node(state)))
        state.update(await loop.run_in_executor(None, lambda: answer_planner_node(state)))

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

    plan = state.get("plan", {}) or {}
    source_nodes = state.get("source_nodes", []) or []

    sources = []
    for node in source_nodes:
        meta = getattr(node, "metadata", {}) or {}
        page = meta.get("page_number") or meta.get("page") or meta.get("page_label")
        section = meta.get("section") or meta.get("header") or ""
        if page:
            sources.append({"page": page, "section": section})

    if sources:
        await websocket.send_text(json.dumps({"type": "sources", "data": sources}))

    await websocket.send_text(json.dumps({
        "type": "plan",
        "data": {
            "mode": plan.get("mode", "direct"),
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
        items = []

        for pdf_file in pdf_dir.glob("*.pdf"):
            collection_id = slugify_filename(pdf_file.name)
            items.append({
                "id": collection_id,
                "title": pdf_file.stem,
                "pdf_url": f"/pdfs/{pdf_file.name}",
                "filename": pdf_file.name,
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