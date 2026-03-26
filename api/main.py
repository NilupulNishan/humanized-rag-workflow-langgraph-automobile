"""
api/main.py

FastAPI application entrypoint.

Endpoints:
  POST /chat           — single-turn REST query
  WS   /ws/{session_id} — WebSocket streaming (tokens arrive live)
  GET  /health         — liveness probe
  GET  /collections    — list available PDF collections
  DELETE /session/{id} — clear session memory

Run:
  uvicorn api.main:app --reload --port 8000
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.chat_router import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("VivoAssist API starting...")

    try:
        from agent.graph import get_graph
        get_graph()
        logger.info("LangGraph graph pre-warmed")
    except Exception as e:
        logger.warning(f"Graph pre-warm failed (non-fatal): {e}")

    yield

    logger.info("VivoAssist API shutting down")


app = FastAPI(
    title="VivoAssist",
    description="Technical support assistant powered by LlamaIndex + LangGraph",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://automobile-rag-frontend-app-ghevamgrawe3f5gw.southeastasia-01.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"

app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VivoAssist"}


@app.get("/session/new")
async def new_session():
    """Generate a fresh session ID for the client."""
    return {"session_id": str(uuid.uuid4())}
