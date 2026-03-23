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

# ─── Node status map ──────────────────────────────────────────────────────────
# Single source of truth — served via GET /pipeline-status
# and used by WebSocket status events.

NODE_STATUS = {
    "query_understanding": {"icon": "🧠", "label": "Understanding question…"},
    "memory_read":         {"icon": "💭", "label": "Reading session context…"},
    "skip_retrieval":      {"icon": "⏭️",  "label": "Preparing clarification…"},
    "retriever":           {"icon": "🔍", "label": "Searching the manual…"},
    "answer_planner":      {"icon": "📋", "label": "Planning the answer…"},
    "web_search":          {"icon": "🌐", "label": "Searching the web…"},
    "response_renderer":   {"icon": "✍️",  "label": "Writing response…"},
    "memory_write":        {"icon": "💾", "label": "Saving session…"},
    "direct_answer":       {"icon": "💬", "label": "Generating response…"},
    "error":               {"icon": "❌", "label": "Something went wrong"},
}


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    collection: Optional[str] = None


class SourceInfo(BaseModel):
    page: Optional[int | str] = None
    section: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    mode: str
    confidence: float
    sources: list[SourceInfo]
    web_sources: list[dict]
    search_used: bool
    needs_followup: bool


# ─── GET /pipeline-status ─────────────────────────────────────────────────────

@router.get("/pipeline-status")
async def get_pipeline_status():
    """
    Returns the node status map for the frontend.

    Frontend fetches this once on load, then uses it to display
    the correct icon and label for each WebSocket status event.

    Example response:
    {
      "nodes": {
        "retriever":  { "icon": "🔍", "label": "Searching the manual…" },
        "web_search": { "icon": "🌐", "label": "Searching the web…" },
        ...
      }
    }

    WebSocket status event shape:
    {
      "type":  "status",
      "node":  "web_search",
      "done":  false,
      "icon":  "🌐",
      "label": "Searching the web…"
    }
    """
    return {"nodes": NODE_STATUS}


# ─── Shared pipeline runner ───────────────────────────────────────────────────

def _run_pipeline(
    user_input:      str,
    session_id:      str,
    collection_name: str | None,
    on_node_start=None,   # optional callback(node_name: str)
    on_node_done=None,    # optional callback(node_name: str)
) -> dict:
    """
    Runs the full pipeline with optional node lifecycle callbacks.
    Callbacks allow the WebSocket handler to stream live status events.
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

    def run_node(name, fn):
        if on_node_start:
            on_node_start(name)
        result = fn(state)
        result.pop("collection_name", None)
        state.update(result)
        if on_node_done:
            on_node_done(name)

    # ── Node 1 ────────────────────────────────────────────────────────────
    run_node("query_understanding", query_understanding_node)

    analysis            = state.get("analysis", {}) or {}
    intent              = analysis.get("intent", "")
    needs_clarification = analysis.get("needs_clarification", False)

    # ── Route ─────────────────────────────────────────────────────────────
    if intent == "general":
        from agent.graph import direct_answer_node
        run_node("direct_answer", direct_answer_node)

    elif intent == "this_car_vs_another_comparison":
        run_node("web_search", web_search_node)

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
        state["raw_answer"]           = ""
        state["source_nodes"]         = []
        state["retrieval_successful"] = False

    else:
        run_node("memory_read",    memory_read_node)
        run_node("retriever",      retriever_node)
        run_node("answer_planner", answer_planner_node)

        plan        = state.get("plan", {}) or {}
        mode        = plan.get("mode", "direct")
        confidence  = plan.get("confidence", 1.0)
        search_used = state.get("search_used", False)

        if not search_used and (
            mode in ("web_search_needed", "escalate") or confidence < 0.35
        ):
            run_node("web_search", web_search_node)

    # ── Renderer ──────────────────────────────────────────────────────────
    if not state.get("final_response"):
        run_node("response_renderer", response_renderer_node)

    # ── Memory write ──────────────────────────────────────────────────────
    run_node("memory_write", memory_write_node)

    return state


def _extract_sources(state: dict) -> tuple[list[SourceInfo], list[dict]]:
    """Returns (manual_sources, web_sources) from state."""
    source_nodes = state.get("source_nodes", []) or []
    search_used  = state.get("search_used", False)
    manual_sources: list[SourceInfo] = []
    web_sources:    list[dict]       = []

    for node in source_nodes:
        meta = getattr(node, "metadata", {}) or {}
        if search_used:
            url   = meta.get("source", "")
            title = meta.get("title", url)
            if url:
                web_sources.append({"url": url, "title": title})
        else:
            page    = meta.get("page_number") or meta.get("page") or meta.get("page_label")
            section = meta.get("section") or meta.get("header") or ""
            if page:
                try:
                    manual_sources.append(SourceInfo(page=int(page), section=section))
                except (ValueError, TypeError):
                    pass

    return manual_sources, web_sources


# ─── POST /chat ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    import asyncio

    session_id = request.session_id or str(uuid.uuid4())

    try:
        state = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_pipeline(
                user_input=request.message,
                session_id=session_id,
                collection_name=request.collection,
            )
        )

        plan                        = state.get("plan", {}) or {}
        manual_sources, web_sources = _extract_sources(state)

        return ChatResponse(
            session_id=session_id,
            response=state.get("final_response", ""),
            mode=plan.get("mode", "direct"),
            confidence=float(plan.get("confidence", 0.5)),
            sources=manual_sources,
            web_sources=web_sources,
            search_used=state.get("search_used", False),
            needs_followup=plan.get("mode") in ("clarify", "troubleshoot"),
        )

    except Exception as e:
        logger.error(f"chat_endpoint error: {e}")
        return ChatResponse(
            session_id=session_id,
            response="I encountered an error. Please try again.",
            mode="error",
            confidence=0.0,
            sources=[],
            web_sources=[],
            search_used=False,
            needs_followup=False,
        )


# ─── WebSocket /ws/{session_id} ───────────────────────────────────────────────
#
# Complete message protocol for the frontend:
#
# ┌─────────────┬──────────────────────────────────────────────────────────────┐
# │ type        │ payload                                                      │
# ├─────────────┼──────────────────────────────────────────────────────────────┤
# │ status      │ { node, done, icon, label }  — one per node start + done    │
# │ token       │ { data: str }                — streams response text        │
# │ sources     │ { data: [{page, section}] }  — manual PDF sources           │
# │ web_sources │ { data: [{url, title}] }     — web search source URLs       │
# │ plan        │ { data: {mode, confidence, likely_goal, search_used} }      │
# │ done        │ { data: null }               — pipeline complete            │
# │ error       │ { data: str }                — pipeline error               │
# └─────────────┴──────────────────────────────────────────────────────────────┘

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

            try:
                await _handle_ws_message(
                    websocket=websocket,
                    user_input=user_input,
                    session_id=session_id,
                    collection_name=collection,
                )
            except Exception as e:
                logger.error(f"WS handler error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", "data": str(e)
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


async def _handle_ws_message(
    websocket:       WebSocket,
    user_input:      str,
    session_id:      str,
    collection_name: str | None,
):
    import asyncio
    from agent.nodes.response_renderer import response_renderer_stream

    loop = asyncio.get_event_loop()

    # ── Node status callbacks ─────────────────────────────────────────────
    # These fire synchronously inside the executor thread and schedule
    # coroutines back on the event loop so websocket.send_text is always
    # called from the async context.

    def _schedule(coro):
        asyncio.run_coroutine_threadsafe(coro, loop)

    async def _send_status(node: str, done: bool):
        info = NODE_STATUS.get(node, {"icon": "⚙️", "label": node})
        await websocket.send_text(json.dumps({
            "type":  "status",
            "node":  node,
            "done":  done,
            "icon":  info["icon"],
            "label": info["label"],
        }))

    def on_node_start(name: str):
        _schedule(_send_status(name, done=False))

    def on_node_done(name: str):
        _schedule(_send_status(name, done=True))

    # ── Run pipeline ──────────────────────────────────────────────────────
    state = await loop.run_in_executor(
        None,
        lambda: _run_pipeline(
            user_input=user_input,
            session_id=session_id,
            collection_name=collection_name,
            on_node_start=on_node_start,
            on_node_done=on_node_done,
        )
    )

    # ── Stream response ───────────────────────────────────────────────────
    full_response = state.get("final_response", "")

    if full_response:
        # Pre-built (direct_answer path) — stream word by word
        words = full_response.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            await websocket.send_text(json.dumps({"type": "token", "data": token}))
    else:
        # Stream live from renderer
        tokens = await loop.run_in_executor(
            None, lambda: list(response_renderer_stream(state))
        )
        full_response = ""
        for token in tokens:
            full_response += token
            await websocket.send_text(json.dumps({"type": "token", "data": token}))

    # ── Send sources ──────────────────────────────────────────────────────
    manual_sources, web_sources = _extract_sources(state)

    if manual_sources:
        await websocket.send_text(json.dumps({
            "type": "sources",
            "data": [s.model_dump() for s in manual_sources],
        }))

    if web_sources:
        await websocket.send_text(json.dumps({
            "type": "web_sources",
            "data": web_sources,
        }))

    # ── Send plan + done ──────────────────────────────────────────────────
    plan = state.get("plan", {}) or {}
    await websocket.send_text(json.dumps({
        "type": "plan",
        "data": {
            "mode":        plan.get("mode", "direct"),
            "confidence":  float(plan.get("confidence", 0.5)),
            "likely_goal": plan.get("likely_goal", ""),
            "search_used": state.get("search_used", False),
        }
    }))

    await websocket.send_text(json.dumps({"type": "done", "data": None}))


# ─── Utility endpoints ────────────────────────────────────────────────────────

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
                "id":       collection_id,
                "title":    pdf_file.stem,
                "pdf_url":  f"/pdfs/{pdf_file.name}",
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