"""
app.py  —  VivoAssist  (LangGraph edition)
"""

from __future__ import annotations
from src.metadata_manager import MetadataManager
from src.storage_manager import StorageManager

import queue
import sys
import threading
import time
import uuid
import logging
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="VivoAssist RAG Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

for k, v in {
    "messages":            [],
    "selected_collection": None,
    "pdf_filename":        None,
    "pdf_page":            1,
    "query_count":         0,
    "session_id":          str(uuid.uuid4()),
    "is_thinking":         False,
    "session_summary":     "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def pdf_exists_on_disk(filename: str) -> bool:
    return bool(filename) and (PDF_DIR / filename).exists()


def render_pdf_viewer_pdfjs(filename: str, page: int, height: int = 720) -> None:
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        st.warning(f"PDF not found: `{filename}` (expected under `{PDF_DIR}`)")
        return

    raw_bytes = pdf_path.read_bytes()
    hex_str = raw_bytes.hex()

    html = f"""
<!doctype html>
<html>
<head><meta charset="utf-8"/></head>
<body>
<div>{filename} · page {int(page)}</div>
<div id="loader">Loading PDF…</div>
<canvas id="cv"></canvas>
<div id="err" style="color:red;font-size:12px;"></div>
<script>
var HEX = "{hex_str}";
var START_PAGE = {int(page)};
function showError(msg) {{{{
  document.getElementById("loader").style.display = "none";
  document.getElementById("err").innerText = "❌ " + msg;
}}}}
function hexToUint8(hex) {{{{
  var len = hex.length / 2;
  var bytes = new Uint8Array(len);
  for (var i = 0; i < len; i++) {{{{ bytes[i] = parseInt(hex.substr(i * 2, 2), 16); }}}}
  return bytes;
}}}}
function loadScript(url, success, fail) {{{{
  var s = document.createElement("script");
  s.src = url; s.onload = success; s.onerror = fail;
  document.head.appendChild(s);
}}}}
function start(pdfjsLib) {{{{
  try {{{{
    var pdfBytes = hexToUint8(HEX);
    pdfjsLib.getDocument({{data: pdfBytes}}).promise.then(function(pdf) {{{{
      var canvas = document.getElementById("cv");
      var ctx = canvas.getContext("2d");
      var p = Math.min(Math.max(START_PAGE, 1), pdf.numPages);
      pdf.getPage(p).then(function(page) {{{{
        var containerWidth = document.body.clientWidth - 20;
        var viewport = page.getViewport({{scale: 1}});
        var displayScale = containerWidth / viewport.width;
        var renderScale = displayScale * 2;
        var renderViewport = page.getViewport({{scale: renderScale}});
        canvas.width = Math.floor(renderViewport.width);
        canvas.height = Math.floor(renderViewport.height);
        canvas.style.width = Math.floor(containerWidth) + "px";
        canvas.style.height = "auto";
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        page.render({{canvasContext: ctx, viewport: renderViewport}}).promise.then(function() {{{{
          document.getElementById("loader").style.display = "none";
        }}}}).catch(function(e) {{{{ showError("Render error: " + e.message); }}}});
      }}}}).catch(function(e) {{{{ showError("Page load error: " + e.message); }}}});
    }}}}).catch(function(e) {{{{ showError("PDF load failed: " + e.message); }}}});
  }}}} catch(e) {{{{ showError("Render error: " + e.message); }}}}
}}}}
loadScript(
  "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.min.js",
  function() {{{{ start(pdfjsLib); }}}},
  function() {{{{
    loadScript(
      "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js",
      function() {{{{ start(pdfjsLib); }}}},
      function() {{{{ showError("Failed to load PDF.js"); }}}}
    );
  }}}}
);
</script>
</body>
</html>
"""
    st_html(html, height=height, scrolling=False)


def render_source_pills(nodes, *, key_prefix: str) -> None:
    if not nodes:
        return
    mm = MetadataManager()
    pages = mm.extract_pages_from_nodes(nodes)
    if not pages:
        return
    ranges = mm.merge_consecutive_pages(pages)
    fname = mm.extract_filename_from_nodes(nodes)
    if not fname:
        return
    per_row = 6
    for r in range(0, len(ranges), per_row):
        row = ranges[r: r + per_row]
        cols = st.columns(len(row))
        for i, (start, end) in enumerate(row):
            label = mm.format_page_range(start, end)
            k = f"{key_prefix}_p_{start}_{end}"
            if cols[i].button(f"• {label}", key=k, use_container_width=True):
                if pdf_exists_on_disk(fname):
                    st.session_state.pdf_filename = fname
                    st.session_state.pdf_page = int(start)
                    st.rerun()
                else:
                    st.warning(f"Source PDF `{fname}` not found in `{PDF_DIR}`.")


@st.cache_resource
def get_storage():
    return StorageManager()


@st.cache_resource
def get_collections():
    return get_storage().list_collections()


BG      = "#f6f8ff"
SIDEBAR = "#ffffff"
PANEL   = "#ffffff"
TEXT    = "#0b1b2b"
BORDER  = "#cfe0ff"
ACCENT  = "#2563eb"
CHIP    = "#f1f5ff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;500;600&display=swap');
:root{{--bg:{BG};--sidebar:{SIDEBAR};--panel:{PANEL};--text:{TEXT};--border:{BORDER};--accent:{ACCENT};--chip:{CHIP};}}
#MainMenu,footer{{visibility:hidden;}}
.stApp{{background-color:var(--bg);}}
.block-container{{padding:1.4rem 2rem 2rem 2rem !important;background-color:var(--bg);}}
html,body,[class*="css"]{{font-family:'Sora',sans-serif;color:var(--text);}}
[data-testid="stSidebar"]{{background:var(--sidebar) !important;border-right:1px solid var(--border);}}
[data-testid="stSidebar"] *{{color:var(--text) !important;}}
[data-testid="collapsedControl"]{{display:block !important;visibility:visible !important;opacity:1 !important;z-index:9999 !important;}}
[data-testid="collapsedControl"] button{{border:1px solid var(--border) !important;background:var(--sidebar) !important;}}
header[data-testid="stHeader"]{{background:transparent !important;}}
.stButton > button{{background-color:var(--panel) !important;color:var(--text) !important;
  border:1px solid var(--border) !important;font-family:'JetBrains Mono',monospace !important;
  font-size:11px !important;transition:all 0.14s;}}
.stButton > button:hover{{background-color:var(--accent) !important;color:#fff !important;border-color:var(--accent) !important;}}
[data-testid="stChatMessage"] > div > div{{padding:0px 10px 0px 10px !important;}}
[data-testid="stChatMessage"]{{border-radius:10px;margin-bottom:6px;background-color:var(--panel);border:1px solid var(--border);}}
[data-testid="stChatInput"]{{background-color:var(--panel);border:2px solid var(--border) !important;border-radius:8px;}}
[data-testid="stChatInput"] input{{background-color:var(--panel);color:var(--text) !important;}}
[data-testid="stChatInput"] input::placeholder{{color:var(--text) !important;opacity:0.6;}}
.coll-badge{{display:inline-block;background:var(--chip);border:1px solid var(--border);color:var(--text);
  font-family:'JetBrains Mono',monospace;font-size:10px;padding:2px 8px;border-radius:4px;
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px;}}
.stat-row{{display:flex;gap:8px;margin-bottom:14px;}}
.stat-card{{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:9px 10px;text-align:center;}}
.stat-num{{font-family:'JetBrains Mono',monospace;font-size:19px;font-weight:600;color:var(--text);line-height:1;}}
.stat-label{{font-size:9px;color:var(--text);letter-spacing:.06em;text-transform:uppercase;margin-top:3px;opacity:0.75;}}
.empty-pdf{{display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:460px;gap:14px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;
  letter-spacing:.05em;border:1px dashed var(--border);border-radius:12px;background:var(--panel);}}
.empty-pdf .ei{{font-size:40px;opacity:.5;}}
div[data-testid="stHorizontalBlock"] .stButton > button{{font-size:5px !important;padding:4px 6px !important;
  line-height:1.1 !important;min-height:0px !important;height:auto !important;border-radius:25px !important;
  background-color:#f2f6ff !important;color:#000000 !important;border:1px solid #5682e8 !important;}}
div[data-testid="stHorizontalBlock"] .stButton > button:hover{{background-color:#5682e8 !important;color:#ffffff !important;border-color:#5682e8 !important;}}
div[data-testid="stHorizontalBlock"] .stButton > button div{{font-size:12px !important;}}
.status-pill{{
  display:inline-flex;align-items:center;gap:7px;
  background:#eef3ff;border:1px solid {BORDER};border-radius:20px;
  padding:4px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;
  color:{TEXT};margin:2px 0;opacity:.9;
  animation:blink 1.4s ease-in-out infinite;
}}
.status-pill.done{{animation:none;border-color:#a3c4fb;color:{ACCENT};background:#f0f5ff;}}
.status-pill.error{{animation:none;border-color:#fca5a5;color:#b91c1c;background:#fff1f1;}}
@keyframes blink{{0%,100%{{opacity:.9;}}50%{{opacity:.45;}}}}
.mode-pill{{
  display:inline-block;padding:1px 9px;border-radius:10px;
  font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;
  background:var(--chip);border:1px solid {BORDER};color:{ACCENT};
  margin-left:6px;vertical-align:middle;
}}
</style>
""", unsafe_allow_html=True)


# ─── Pipeline status config ───────────────────────────────────────────────────
NODE_STATUS = {
    "query_understanding": ("🧠", "Understanding question…"),
    "memory_read":         ("💭", "Reading session context…"),
    "skip_retrieval":      ("⏭️",  "Preparing clarification…"),
    "retriever":           ("🔍", "Searching the manual…"),
    "answer_planner":      ("📋", "Planning the answer…"),
    "web_search":          ("🌐", "Searching the web…"),       # FIX 1: added
    "response_renderer":   ("✍️",  "Writing response…"),
    "memory_write":        ("💾", "Saving session…"),
    "direct_answer":       ("💬", "Generating response…"),
    "error":               ("❌", "Something went wrong"),
}


# ─── Background pipeline runner ───────────────────────────────────────────────
def run_pipeline(
    user_input:           str,
    session_id:           str,
    collection_name:      str | None,
    status_q:             queue.Queue,
    token_q:              queue.Queue,
    conversation_history: list | None = None,
):
    try:
        from agent.nodes.query_understanding import query_understanding_node
        from agent.nodes.memory_node import memory_read_node, memory_write_node
        from agent.nodes.retriever_node import retriever_node
        from agent.nodes.answer_planner import answer_planner_node
        from agent.nodes.web_search_node import web_search_node
        from agent.nodes.response_renderer import response_renderer_stream
        from agent.state import AgentState, AnswerPlan

        history = conversation_history or []
        history_for_llm = []
        for m in history[-10:]:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                history_for_llm.append({"role": role, "content": content})
        history_for_llm.append({"role": "user", "content": user_input})

        state: AgentState = {
            "user_input":      user_input,
            "session_id":      session_id,
            "collection_name": collection_name or "",
            "messages":        history_for_llm,
        }

        def run_node(name, fn):
            status_q.put({"node": name, "done": False})
            result = fn(state)
            result.pop("collection_name", None)
            state.update(result)
            status_q.put({"node": name, "done": True})

        # ── Node 1: query understanding ───────────────────────────────────
        run_node("query_understanding", query_understanding_node)

        # ── FIX 2: full intent-based routing ─────────────────────────────
        analysis            = state.get("analysis", {}) or {}
        intent              = analysis.get("intent", "")
        needs_clarification = analysis.get("needs_clarification", False)

        if intent == "general":
            from agent.graph import direct_answer_node
            run_node("direct_answer", direct_answer_node)

        elif intent == "this_car_vs_another_comparison":
            run_node("web_search", web_search_node)

        elif needs_clarification:
            status_q.put({"node": "skip_retrieval", "done": False})
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
            status_q.put({"node": "skip_retrieval", "done": True})

        else:
            # Normal pipeline
            run_node("memory_read",    memory_read_node)
            run_node("retriever",      retriever_node)
            run_node("answer_planner", answer_planner_node)

            # Post-planner web search fallback
            plan        = state.get("plan", {}) or {}
            mode        = plan.get("mode", "direct")
            confidence  = plan.get("confidence", 1.0)
            search_used = state.get("search_used", False)
            if not search_used and (
                mode in ("web_search_needed", "escalate") or confidence < 0.35
            ):
                run_node("web_search", web_search_node)

        # ── Streaming renderer ────────────────────────────────────────────
        # Skip renderer if direct_answer already set final_response
        if not state.get("final_response"):
            status_q.put({"node": "response_renderer", "done": False})
            full_response = ""
            for token in response_renderer_stream(state):
                token_q.put(token)
                full_response += token
            state["final_response"] = full_response
            state["response_ready"] = True
            status_q.put({"node": "response_renderer", "done": True})
        else:
            full_response = state.get("final_response", "")
            # Stream the pre-built response token by token
            for token in full_response:
                token_q.put(token)

        # ── Memory write ──────────────────────────────────────────────────
        run_node("memory_write", memory_write_node)

        # ── Build metadata ────────────────────────────────────────────────
        plan         = state.get("plan", {}) or {}
        source_nodes = state.get("source_nodes", []) or []
        session      = state.get("session")
        search_used  = state.get("search_used", False)

        sources     = []
        web_sources = []
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
                        sources.append({"page": int(page), "section": section})
                    except (ValueError, TypeError):
                        pass

        session_summary = ""
        if session and hasattr(session, "to_context_string"):
            session_summary = session.to_context_string()

        token_q.put({"DONE": True, "meta": {
            "mode":            plan.get("mode", "direct"),
            "confidence":      plan.get("confidence", 0.5),
            "likely_goal":     plan.get("likely_goal", ""),
            "sources":         sources,
            "source_nodes":    source_nodes,
            "web_sources":     web_sources,
            "session_summary": session_summary,
            "full_response":   full_response,
            "collection":      collection_name,
            "search_used":     search_used,
        }})

    except Exception as e:
        logger.exception("Pipeline error")
        status_q.put({"node": "error", "done": True, "error": True})
        token_q.put({"ERROR": str(e)})


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## VIVO ASSIST")
    st.markdown("---")

    collections = get_collections()
    if not collections:
        st.error("No collections found.\nRun `python scripts/process_pdfs.py` first.")
        st.stop()

    options = ["— All collections —"] + collections
    idx = 0
    if st.session_state.selected_collection in collections:
        idx = collections.index(st.session_state.selected_collection) + 1

    chosen   = st.selectbox("Collection", options, index=idx)
    selected = None if chosen == "— All collections —" else chosen

    if selected != st.session_state.selected_collection:
        st.session_state.selected_collection = selected
        st.session_state.messages            = []
        st.session_state.pdf_filename        = None
        st.session_state.pdf_page            = 1
        # FIX 3: reset session_id on collection change — clears old memory
        st.session_state.session_id          = str(uuid.uuid4())
        st.session_state.session_summary     = ""
        st.rerun()

    st.markdown("---")

    total_chunks = 0
    try:
        for c in collections:
            total_chunks += get_storage().get_collection_info(c).get("count", 0)
    except Exception:
        pass

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><div class="stat-num">{len(collections)}</div><div class="stat-label">Collections</div></div>
      <div class="stat-card"><div class="stat-num">{total_chunks}</div><div class="stat-label">Chunks</div></div>
      <div class="stat-card"><div class="stat-num">{st.session_state.query_count}</div><div class="stat-label">Queries</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.session_summary:
        st.markdown("**🧠 Session context**")
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;'
            f'color:{TEXT};background:{CHIP};border:1px solid {BORDER};'
            f'border-radius:6px;padding:8px 10px;line-height:1.6;">'
            f'{st.session_state.session_summary}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.pdf_filename    = None
        st.session_state.pdf_page        = 1
        st.session_state.query_count     = 0
        st.session_state.session_id      = str(uuid.uuid4())
        st.session_state.session_summary = ""
        st.rerun()

    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:{TEXT};margin-top:8px;opacity:.85;">'
        f'● inline PDF viewer active</div>',
        unsafe_allow_html=True,
    )
    st.caption("LlamaIndex · LangGraph · ChromaDB · Azure OpenAI")


# ─── Main columns ─────────────────────────────────────────────────────────────
col_chat, col_pdf = st.columns([1, 1], gap="large")

with col_chat:
    st.markdown("### Ask a question")

    CHAT_HEIGHT = 650
    chat_area = st.container(height=CHAT_HEIGHT)

    with chat_area:
        for mi, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    meta = msg.get("meta", {})
                    coll = meta.get("collection") or msg.get("collection")
                    mode = meta.get("mode", "")

                    badge_html = ""
                    if coll:
                        badge_html += f'<span class="coll-badge">📁 {coll}</span>'
                    if mode:
                        badge_html += f'<span class="mode-pill">{mode}</span>'
                    if badge_html:
                        st.markdown(badge_html, unsafe_allow_html=True)

                    st.markdown(msg["content"])

                    # Source pills for manual answers
                    nodes = meta.get("source_nodes") or msg.get("nodes", [])
                    if nodes and not meta.get("search_used"):
                        render_source_pills(nodes, key_prefix=f"hist_{mi}")

                    # Web source links for web search answers
                    web_sources = meta.get("web_sources", [])
                    if web_sources:
                        st.markdown("**Sources:**")
                        for ws in web_sources:
                            st.markdown(f"- [{ws['title']}]({ws['url']})")
                else:
                    st.markdown(msg["content"])

        tail = st.empty()

    query = st.chat_input(
        "Ask anything about your PDFs…",
        disabled=st.session_state.is_thinking,
    )

    if query and not st.session_state.is_thinking:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.is_thinking = True

        with tail.container():
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                coll_label = st.session_state.selected_collection
                if coll_label:
                    st.markdown(
                        f'<span class="coll-badge">📁 {coll_label}</span>',
                        unsafe_allow_html=True,
                    )

                status_ph  = st.empty()
                response_ph = st.empty()
                pills_ph   = st.empty()

                status_q: queue.Queue = queue.Queue()
                token_q:  queue.Queue = queue.Queue()

                thread = threading.Thread(
                    target=run_pipeline,
                    args=(
                        query,
                        st.session_state.session_id,
                        st.session_state.selected_collection,
                        status_q,
                        token_q,
                        list(st.session_state.messages),
                    ),
                    daemon=True,
                )
                thread.start()

                accumulated    = ""
                final_meta     = {}
                pipeline_error = None

                while True:
                    try:
                        while True:
                            s     = status_q.get_nowait()
                            node  = s["node"]
                            done  = s.get("done", False)
                            error = s.get("error", False)
                            icon, label = NODE_STATUS.get(node, ("⚙️", node))
                            css = "status-pill"
                            if error:
                                css += " error"
                            elif done and node not in ("response_renderer",):
                                css += " done"
                            status_ph.markdown(
                                f'<div class="{css}">{icon} {label}</div>',
                                unsafe_allow_html=True,
                            )
                    except queue.Empty:
                        pass

                    done_signal = False
                    try:
                        while True:
                            item = token_q.get_nowait()
                            if isinstance(item, dict):
                                if item.get("DONE"):
                                    final_meta  = item["meta"]
                                    done_signal = True
                                    break
                                elif item.get("ERROR"):
                                    pipeline_error = item["ERROR"]
                                    done_signal    = True
                                    break
                            else:
                                accumulated += item
                                response_ph.markdown(accumulated + "▌")
                    except queue.Empty:
                        pass

                    if done_signal:
                        break
                    time.sleep(0.02)

                thread.join(timeout=5)
                status_ph.empty()

                if pipeline_error:
                    response_ph.error(
                        f"Something went wrong: {pipeline_error}\n\nPlease try rephrasing."
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ {pipeline_error}",
                        "meta": {},
                    })
                else:
                    response_ph.markdown(accumulated)

                    mode = final_meta.get("mode", "")
                    conf = final_meta.get("confidence", 0)
                    if mode:
                        st.markdown(
                            f'<span class="mode-pill">{mode}</span>'
                            f'<span style="font-size:10px;color:{TEXT};opacity:.6;margin-left:6px;">'
                            f'{conf:.0%}</span>',
                            unsafe_allow_html=True,
                        )

                    source_nodes = final_meta.get("source_nodes", [])
                    web_sources  = final_meta.get("web_sources", [])
                    search_used  = final_meta.get("search_used", False)

                    with pills_ph.container():
                        if not search_used:
                            render_source_pills(
                                source_nodes,
                                key_prefix=f"live_{st.session_state.query_count}",
                            )
                        if web_sources:
                            st.markdown("**Sources:**")
                            for ws in web_sources:
                                st.markdown(f"- [{ws['title']}]({ws['url']})")

                    if source_nodes and not search_used:
                        mm = MetadataManager()
                        pages = mm.extract_pages_from_nodes(source_nodes)
                        fname = mm.extract_filename_from_nodes(source_nodes)
                        if pages and fname and pdf_exists_on_disk(fname):
                            st.session_state.pdf_filename = fname
                            st.session_state.pdf_page     = int(pages[0])

                    if final_meta.get("session_summary"):
                        st.session_state.session_summary = final_meta["session_summary"]

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": accumulated,
                        "meta":    final_meta,
                    })
                    st.session_state.query_count += 1

                st.session_state.is_thinking = False

        st.rerun()


# ─── PDF Viewer ───────────────────────────────────────────────────────────────
with col_pdf:
    st.markdown("### 📄 Source document")

    fname = st.session_state.pdf_filename
    page  = int(st.session_state.pdf_page or 1)

    if fname:
        col_info, col_jump = st.columns([3, 1])
        with col_info:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;'
                f'color:{TEXT};padding-top:6px;">'
                f'<span style="color:{ACCENT};">●</span>&nbsp;{fname}'
                f'&nbsp;·&nbsp;page {page}</div>',
                unsafe_allow_html=True,
            )
        with col_jump:
            new_page = st.number_input(
                "page", min_value=1, value=page, step=1,
                label_visibility="collapsed",
                key=f"pjump_{fname}_{page}",
            )
            if int(new_page) != page:
                st.session_state.pdf_page = int(new_page)
                st.rerun()

        render_pdf_viewer_pdfjs(fname, page, height=720)
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;'
            f'color:{TEXT};margin-top:8px;opacity:.7;">Inline PDF viewer</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="empty-pdf">
          <div class="ei">📄</div>
          <div>Ask a question — the source PDF will appear here</div>
        </div>""", unsafe_allow_html=True)