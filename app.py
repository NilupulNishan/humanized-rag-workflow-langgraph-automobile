"""
Streamlit RAG interface — chat UI + embedded PDF viewer (no cross-origin issues).

The PDF is read from disk, base64-encoded, and injected as a data URI directly
into a components.v1.html() block — same origin as Streamlit, Chrome never
blocks it.

Run:
    streamlit run app.py
"""

import sys
import base64
import logging
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="PDF·RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.storage_manager import StorageManager

# ── Use optimized retriever ──────────────────────────────────────────────────
# StreamResult is iterable → plugs directly into st.write_stream()
# source_nodes live on the StreamResult object after stream completes —
# no second API call needed.
from src.retriever import SmartRetriever, MultiCollectionRetriever   # swap if needed
# from retriever_optimized import SmartRetriever, MultiCollectionRetriever, StreamResult
# ─────────────────────────────────────────────────────────────────────────────

from src.metadata_manager import MetadataManager
from pdf_server import get_viewer_url, SERVER_PORT, start_server_background

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Start pdf_server so citation pills (open-in-new-tab) still work
@st.cache_resource
def _boot_pdf_server():
    start_server_background()

_boot_pdf_server()

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def pdf_to_b64(filename: str) -> str | None:
    """Read PDF from disk and return base64 string, or None if missing."""
    path = PDF_DIR / filename
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def render_pdf_viewer(filename: str, page: int, height: int = 700) -> None:
    """
    Render a PDF viewer entirely inside Streamlit using a base64 data URI.
    No external server request — Chrome cannot block it.
    """
    b64 = pdf_to_b64(filename)
    if b64 is None:
        st.warning(f"PDF not found: `{filename}`")
        return

    viewer_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #00072c;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
  }}
  .bar {{
    display: flex; align-items: center; gap: 12px;
    padding: 7px 14px;
    background: #001c54; border-bottom: 1px solid #0e6ba8;
    flex-shrink: 0; min-height: 40px;
  }}
  .logo  {{ font-size:10px; font-weight:600; letter-spacing:.12em; color:#a5e1f9; text-transform:uppercase; }}
  .fname {{ font-size:11px; color:#a5e1f9; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .pbadge {{
    font-size:10px; color:#0e6ba8;
    background:#a5e1f9; border:1px solid #0e6ba8;
    padding:2px 8px; border-radius:4px; white-space:nowrap;
  }}
  .nav {{ display:flex; align-items:center; gap:6px; }}
  .nav input {{
    width:48px; background:#00072c; border:1px solid #0e6ba8;
    border-radius:4px; color:#a5e1f9; font-family:inherit;
    font-size:11px; padding:2px 6px; text-align:center; outline:none;
  }}
  .nav input:focus {{ border-color:#a5e1f9; }}
  .nav button {{
    background:#0a2471; color:#a5e1f9; border:1px solid #0e6ba8;
    border-radius:4px; font-family:inherit; font-size:10px; font-weight:600;
    padding:2px 8px; cursor:pointer; transition:all .12s;
  }}
  .nav button:hover {{ background:#0e6ba8; color:#a5e1f9; }}
  iframe {{ flex:1; width:100%; border:none; display:block; }}
</style>
</head>
<body>
<div class="bar">
  <span class="logo">PDF·RAG</span>
  <span class="fname" title="{filename}">{filename}</span>
  <span class="pbadge" id="pbadge">Page {page}</span>
  <div class="nav">
    <input id="pg" type="number" min="1" value="{page}">
    <button onclick="jump()">Go</button>
  </div>
</div>
<iframe id="pdf"
  src="data:application/pdf;base64,{b64}#page={page}"
  title="{filename}">
</iframe>
<script>
  document.getElementById('pg').addEventListener('input', function() {{
    document.getElementById('pbadge').textContent = 'Page ' + (this.value || '?');
  }});
  function jump() {{
    var p = parseInt(document.getElementById('pg').value, 10);
    if (!p || p < 1) return;
    document.getElementById('pbadge').textContent = 'Page ' + p;
    document.getElementById('pdf').src =
      'data:application/pdf;base64,{b64}#page=' + p;
  }}
  document.getElementById('pg').addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') jump();
  }});
  document.addEventListener('keydown', function(e) {{
    var inp = document.getElementById('pg');
    var cur = parseInt(inp.value, 10) || 1;
    if (e.key === '[' && cur > 1) {{ inp.value = cur - 1; jump(); }}
    if (e.key === ']')            {{ inp.value = cur + 1; jump(); }}
  }});
</script>
</body>
</html>"""

    st.components.v1.html(viewer_html, height=height, scrolling=False)


# ─── Source pills builder (reused in both history render and new response) ────

def build_pills_html(nodes: list, mm: MetadataManager) -> tuple[str, str, int]:
    """
    Build source pill HTML from nodes.
    Returns (pills_html, filename, first_page).
    Returns ("", "", 0) if no page info.
    """
    pages  = mm.extract_pages_from_nodes(nodes)
    if not pages:
        return "", "", 0

    ranges = mm.merge_consecutive_pages(pages)
    fname  = mm.extract_filename_from_nodes(nodes)

    pills = '<div style="margin-top:8px;line-height:2.4;">'
    for start, end in ranges:
        label = mm.format_page_range(start, end)
        url   = get_viewer_url(fname, start)
        pills += (
            f'<a class="source-pill" href="{url}" target="_blank">'
            f'<span class="dot"></span>{label}</a>'
        )
    pills += "</div>"
    return pills, fname, pages[0]


# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource
def get_storage():
    return StorageManager()

@st.cache_resource
def get_collections():
    return get_storage().list_collections()

@st.cache_resource
def get_retriever(collection_name: str | None):
    if collection_name:
        return SmartRetriever(collection_name, verbose=False)
    return MultiCollectionRetriever(verbose=False)


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; color: #a5e1f9; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.4rem 2rem 2rem 2rem !important; background-color: #00072c; }

/* Main background */
.stApp {
    background-color: #00072c;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #001c54 !important;
    border-right: 1px solid #0e6ba8;
}
[data-testid="stSidebar"] * { color: #a5e1f9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a5e1f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: .05em;
}

/* Selectbox styling */
[data-testid="stSelectbox"] {
    background-color: #0a2471;
    border-radius: 8px;
    border: 1px solid #0e6ba8;
}
[data-testid="stSelectbox"] > div {
    background-color: #0a2471;
}
[data-testid="stSelectbox"] input {
    background-color: #0a2471;
    color: #a5e1f9;
}

/* Button styling */
.stButton > button {
    background-color: #0a2471 !important;
    color: #a5e1f9 !important;
    border: 1px solid #0e6ba8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    transition: all 0.14s;
}
.stButton > button:hover {
    background-color: #0e6ba8 !important;
    color: #a5e1f9 !important;
    border-color: #a5e1f9 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 6px;
    background-color: #0a2471;
    border: 1px solid #0e6ba8;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #a5e1f9;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #0a2471;
    border: 2px solid #0e6ba8 !important;
    border-radius: 8px;
}
[data-testid="stChatInput"] input {
    background-color: #0a2471;
    color: #a5e1f9 !important;
}
[data-testid="stChatInput"] input::placeholder {
    color: #a5e1f9 !important;
    opacity: 0.7;
}

.source-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #001c54; border: 1px solid #0e6ba8;
    border-radius: 20px; padding: 4px 12px 4px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: #a5e1f9; text-decoration: none;
    margin: 3px 4px 3px 0; cursor: pointer; transition: all .14s;
}
.source-pill:hover { background: #0a2471; border-color: #a5e1f9; color: #a5e1f9; }
.source-pill .dot { width:5px; height:5px; border-radius:50%; background:#0e6ba8; flex-shrink:0; }

.coll-badge {
    display: inline-block; background: #001c54; border: 1px solid #0e6ba8;
    color: #a5e1f9; font-family: 'JetBrains Mono', monospace; font-size: 10px;
    padding: 2px 8px; border-radius: 4px; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: 6px;
}

.stat-row { display:flex; gap:8px; margin-bottom:14px; }
.stat-card {
    flex:1; background:#0a2471; border:1px solid #0e6ba8;
    border-radius:8px; padding:9px 10px; text-align:center;
}
.stat-num  { font-family:'JetBrains Mono',monospace; font-size:19px; font-weight:600; color:#a5e1f9; line-height:1; }
.stat-label { font-size:9px; color:#a5e1f9; letter-spacing:.06em; text-transform:uppercase; margin-top:3px; opacity:0.8; }

.empty-pdf {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; height:460px; gap:14px;
    color:#a5e1f9; font-family:'JetBrains Mono',monospace; font-size:12px;
    letter-spacing:.05em; border:1px dashed #0e6ba8; border-radius:12px;
    background: #0a2471;
}
.empty-pdf .ei { font-size:40px; opacity:.5; }

/* Scrollbar styling */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #001c54; }
::-webkit-scrollbar-thumb { background: #a5e1f9; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #0e6ba8; }

/* Number input */
[data-testid="stNumberInput"] input {
    background-color: #0a2471; border: 1px solid #0e6ba8; color: #a5e1f9;
}
[data-testid="stNumberInput"] button {
    background-color: #0a2471; color: #a5e1f9; border: 1px solid #0e6ba8;
}
[data-testid="stNumberInput"] button:hover {
    background-color: #0e6ba8; color: #a5e1f9;
}

[data-testid="stMarkdownContainer"] { color: #a5e1f9; }
h1, h2, h3, h4, h5, h6 { color: #a5e1f9 !important; }
.stRadio > div { color: #a5e1f9; }
.stCheckbox > div { color: #a5e1f9; }
[data-testid="stTabs"] { color: #a5e1f9; }
[data-testid="stTabs"] button { color: #a5e1f9; }
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #0a2471; border-bottom: 2px solid #0e6ba8;
}
[data-testid="stExpander"] {
    background-color: #0a2471; border: 1px solid #0e6ba8;
}
[data-testid="stExpander"] summary { color: #a5e1f9; }
.stAlert { background-color: #0a2471; border: 1px solid #0e6ba8; color: #a5e1f9; }
.stProgress > div > div { background-color: #0e6ba8; }
[data-testid="stDataFrame"] { color: #a5e1f9; }
[data-testid="stDataFrame"] th { background-color: #001c54; color: #a5e1f9; }
[data-testid="stDataFrame"] td { background-color: #0a2471; color: #a5e1f9; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

for k, v in {
    "messages": [],
    "selected_collection": None,
    "pdf_filename": None,
    "pdf_page": 1,
    "query_count": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## PDF·RAG")
    st.markdown("---")

    collections = get_collections()
    if not collections:
        st.error("No collections found.\nRun `process_pdfs.py` first.")
        st.stop()

    options = ["— All collections —"] + collections
    idx = 0
    if st.session_state.selected_collection in collections:
        idx = collections.index(st.session_state.selected_collection) + 1

    chosen = st.selectbox("Collection", options, index=idx)
    selected = None if chosen == "— All collections —" else chosen

    if selected != st.session_state.selected_collection:
        st.session_state.selected_collection = selected
        st.session_state.messages = []
        st.session_state.pdf_filename = None
        st.session_state.pdf_page = 1
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
      <div class="stat-card">
        <div class="stat-num">{len(collections)}</div>
        <div class="stat-label">Collections</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{total_chunks}</div>
        <div class="stat-label">Chunks</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{st.session_state.query_count}</div>
        <div class="stat-label">Queries</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pdf_filename = None
        st.session_state.pdf_page = 1
        st.session_state.query_count = 0
        st.rerun()

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#a5e1f9;margin-top:8px;">
      ● PDF server · port {SERVER_PORT}
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("LlamaIndex · ChromaDB · Azure OpenAI")


# ─── Main columns ─────────────────────────────────────────────────────────────

col_chat, col_pdf = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Chat
# ══════════════════════════════════════════════════════════════════════════════

with col_chat:
    st.markdown("### 💬 Ask a question")

    # ── Render chat history ───────────────────────────────────────────────────
    mm = MetadataManager()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if msg.get("collection"):
                    st.markdown(
                        f'<span class="coll-badge">📁 {msg["collection"]}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(msg["content"])

                nodes = msg.get("nodes", [])
                if nodes:
                    pills, _, _ = build_pills_html(nodes, mm)
                    if pills:
                        st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    # ── Handle new query ──────────────────────────────────────────────────────
    query = st.chat_input("Ask anything about your PDFs…")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                retriever  = get_retriever(st.session_state.selected_collection)
                is_multi   = isinstance(retriever, MultiCollectionRetriever)

                if is_multi:
                    # ── Multi-collection: blocking (streaming not yet supported) ──
                    with st.spinner("Searching across collections…"):
                        response = retriever.query_best(query)

                    coll_label = response.collection_name

                    if response.retrieval_successful:
                        if coll_label:
                            st.markdown(
                                f'<span class="coll-badge">📁 {coll_label}</span>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(response.answer)
                        answer = response.answer
                        nodes  = response.source_nodes
                    else:
                        err = response.error_message or "Unknown error"
                        st.error(f"Retrieval failed: {err}")
                        st.session_state.messages.append({
                            "role": "assistant", "content": f"⚠️ {err}", "nodes": [],
                        })
                        st.rerun()

                else:
                    # ── Single collection: STREAMING ──────────────────────────
                    #
                    # Flow:
                    #   retriever.stream(query)
                    #     → embed query         (cold: ~900ms / warm: 0ms cache)
                    #     → vector search+merge (~200ms)  ← source_nodes ready
                    #     → LLM starts generating
                    #   st.write_stream(result)
                    #     → renders tokens live as they arrive
                    #     → returns the full concatenated string when done
                    #   result.source_nodes
                    #     → already populated, zero extra API call
                    #
                    # User sees: first word in ~1s (warm) or ~1.5s (cold)
                    # instead of blank screen for 8-10s.
                    # ─────────────────────────────────────────────────────────

                    coll_label = st.session_state.selected_collection

                    if coll_label:
                        st.markdown(
                            f'<span class="coll-badge">📁 {coll_label}</span>',
                            unsafe_allow_html=True,
                        )

                    # Call stream() — retrieval runs immediately,
                    # LLM generation starts, tokens flow into write_stream
                    result = retriever.stream(query)

                    if result.failed:
                        st.error("Retrieval failed")
                        st.rerun()

                    # st.write_stream() accepts any generator.
                    # It renders each token live and returns the full string.
                    answer = st.write_stream(result)

                    # source_nodes were populated during retrieval (before LLM
                    # even started) — reading them here costs zero extra calls.
                    nodes = result.source_nodes

                # ── Source pills (same for both paths) ───────────────────────
                pills, fname, first_page = build_pills_html(nodes, mm)
                if pills:
                    st.markdown(pills, unsafe_allow_html=True)
                    st.session_state.pdf_filename = fname
                    st.session_state.pdf_page     = first_page

                # ── Save to history ───────────────────────────────────────────
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "nodes": nodes,
                    "collection": coll_label,
                })
                st.session_state.query_count += 1

            except Exception as e:
                logger.exception("Query error")
                st.error(f"Error: {e}")

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — PDF Viewer (base64 data URI, no cross-origin request)
# ══════════════════════════════════════════════════════════════════════════════

with col_pdf:
    st.markdown("### 📄 Source document")

    fname = st.session_state.pdf_filename
    page  = st.session_state.pdf_page or 1

    if fname:
        col_info, col_jump = st.columns([3, 1])
        with col_info:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;'
                f'color:#a5e1f9;padding-top:6px;">'
                f'<span style="color:#0e6ba8;">●</span>&nbsp;{fname}'
                f'&nbsp;·&nbsp;<span style="color:#a5e1f9;">page {page}</span></div>',
                unsafe_allow_html=True,
            )
        with col_jump:
            new_page = st.number_input(
                "page", min_value=1, value=page, step=1,
                label_visibility="collapsed", key=f"pjump_{fname}",
            )
            if new_page != page:
                st.session_state.pdf_page = new_page
                st.rerun()

        render_pdf_viewer(fname, page, height=700)

        viewer_url = get_viewer_url(fname, page)
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" '
            f'style="font-family:JetBrains Mono,monospace;font-size:11px;'
            f'color:#a5e1f9;text-decoration:none;border:1px solid #0e6ba8;padding:4px 12px;border-radius:4px;">↗ open in new tab</a>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="empty-pdf">
          <div class="ei">📄</div>
          <div>Ask a question — the source PDF will appear here</div>
        </div>
        """, unsafe_allow_html=True)