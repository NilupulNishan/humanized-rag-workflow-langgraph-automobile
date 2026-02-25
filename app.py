"""
Streamlit RAG interface — chat UI + embedded PDF viewer side-by-side.

Run:
    streamlit run app.py
"""

import sys
import time
import logging
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from streamlit.components.v1 import html as st_html

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PDF·RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports (avoid blocking on startup) ─────────────────────────────────
from src.storage_manager import StorageManager
from src.retriever import SmartRetriever, MultiCollectionRetriever
from src.metadata_manager import MetadataManager
from pdf_server import start_server_background, get_viewer_url, SERVER_PORT

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Start PDF server once per process ────────────────────────────────────────
@st.cache_resource
def boot_pdf_server():
    ok = start_server_background()
    return ok

boot_pdf_server()

# ─── Cached resource loaders ─────────────────────────────────────────────────
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

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1a1f2e;
}
[data-testid="stSidebar"] * { color: #c8d0e8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4f8ef7 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 8px;
}

/* ── User bubble ── */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #141828 !important;
}

/* ── Source pills ── */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 4px 12px 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4f8ef7;
    text-decoration: none;
    margin: 3px 4px 3px 0;
    cursor: pointer;
    transition: all 0.15s;
}
.source-pill:hover {
    background: #1a2a4a;
    border-color: #4f8ef7;
    color: #7db3ff;
}
.source-pill .dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #34d399;
    flex-shrink: 0;
}

/* ── Collection badge ── */
.coll-badge {
    display: inline-block;
    background: #0d1a2e;
    border: 1px solid #1e3a5f;
    color: #4f8ef7;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── PDF panel header ── */
.pdf-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 12px 0;
    border-bottom: 1px solid #1a1f2e;
    margin-bottom: 10px;
}
.pdf-header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #4f8ef7;
    text-transform: uppercase;
}
.pdf-header-file {
    font-size: 12px;
    color: #6b7a99;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.pdf-header-page {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #34d399;
    background: #0a1f12;
    border: 1px solid #1a3a22;
    padding: 2px 8px;
    border-radius: 4px;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 420px;
    gap: 12px;
    color: #3a4460;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
    border: 1px dashed #1a2035;
    border-radius: 12px;
}
.empty-state .icon { font-size: 36px; opacity: 0.4; }

/* ── Thinking spinner ── */
.thinking {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #4f8ef7;
    font-size: 13px;
    padding: 8px 0;
}

/* ── Stat card ── */
.stat-row {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}
.stat-card {
    flex: 1;
    background: #0d1118;
    border: 1px solid #1a1f2e;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: #4f8ef7;
    line-height: 1;
}
.stat-label {
    font-size: 10px;
    color: #4a5270;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Scrollable chat area ── */
.chat-scroll {
    overflow-y: auto;
    max-height: calc(100vh - 220px);
}
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],           # [{role, content, nodes, collection}]
        "selected_collection": None,
        "pdf_filename": None,
        "pdf_page": 1,
        "pdf_url": None,
        "query_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PDF·RAG")
    st.markdown("---")

    # Collection picker
    collections = get_collections()

    if not collections:
        st.error("No collections found.\nRun `process_pdfs.py` first.")
        st.stop()

    options = ["— All collections —"] + collections
    idx = 0
    if st.session_state.selected_collection in collections:
        idx = collections.index(st.session_state.selected_collection) + 1

    chosen = st.selectbox(
        "Collection",
        options,
        index=idx,
        help="Pick a specific PDF collection, or search across all of them.",
    )
    selected = None if chosen == "— All collections —" else chosen

    if selected != st.session_state.selected_collection:
        st.session_state.selected_collection = selected
        # Clear chat when switching collection
        st.session_state.messages = []
        st.session_state.pdf_url = None
        st.rerun()

    st.markdown("---")

    # Stats
    total_chunks = 0
    try:
        storage = get_storage()
        for c in collections:
            info = storage.get_collection_info(c)
            total_chunks += info.get("count", 0)
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

    # Clear chat
    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pdf_url = None
        st.session_state.query_count = 0
        st.rerun()

    # PDF server status
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#34d399;margin-top:8px;">
      ● PDF server · port {SERVER_PORT}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built with LlamaIndex + ChromaDB + Azure OpenAI")

# ─── Main layout: chat left, PDF right ───────────────────────────────────────
col_chat, col_pdf = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Chat
# ══════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown("### 💬 Ask a question")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Collection badge
                if msg.get("collection"):
                    st.markdown(
                        f'<span class="coll-badge">📁 {msg["collection"]}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(msg["content"])

                # Source pills
                nodes = msg.get("nodes", [])
                if nodes:
                    mm = MetadataManager()
                    pages  = mm.extract_pages_from_nodes(nodes)
                    ranges = mm.merge_consecutive_pages(pages)
                    fname  = mm.extract_filename_from_nodes(nodes)

                    pills_html = '<div style="margin-top:10px;line-height:2.2;">'
                    for start, end in ranges:
                        page_label = mm.format_page_range(start, end)
                        viewer_url = get_viewer_url(fname, start)
                        pills_html += (
                            f'<a class="source-pill" href="{viewer_url}" target="_blank">'
                            f'<span class="dot"></span>{page_label}</a>'
                        )
                    pills_html += "</div>"
                    st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    query = st.chat_input(
        placeholder="Ask anything about your PDFs…",
        key="chat_input",
    )

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run retriever
        with st.chat_message("assistant"):
            with st.spinner("Searching…"):
                try:
                    retriever = get_retriever(st.session_state.selected_collection)

                    if isinstance(retriever, MultiCollectionRetriever):
                        response = retriever.query_best(query)
                        coll_label = response.collection_name if response.retrieval_successful else None
                    else:
                        response = retriever.query(query)
                        coll_label = st.session_state.selected_collection

                    if response.retrieval_successful:
                        answer = response.answer
                        nodes  = response.source_nodes

                        # Collection badge
                        if coll_label:
                            st.markdown(
                                f'<span class="coll-badge">📁 {coll_label}</span>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(answer)

                        # Source pills + auto-load PDF panel
                        mm = MetadataManager()
                        pages  = mm.extract_pages_from_nodes(nodes)
                        ranges = mm.merge_consecutive_pages(pages)
                        fname  = mm.extract_filename_from_nodes(nodes)

                        if pages:
                            pills_html = '<div style="margin-top:10px;line-height:2.2;">'
                            for start, end in ranges:
                                page_label = mm.format_page_range(start, end)
                                viewer_url = get_viewer_url(fname, start)
                                pills_html += (
                                    f'<a class="source-pill" href="{viewer_url}" target="_blank">'
                                    f'<span class="dot"></span>{page_label}</a>'
                                )
                            pills_html += "</div>"
                            st.markdown(pills_html, unsafe_allow_html=True)

                            # Update PDF panel to first source page
                            st.session_state.pdf_filename = fname
                            st.session_state.pdf_page     = pages[0]
                            st.session_state.pdf_url      = get_viewer_url(fname, pages[0])

                        # Persist to history
                        st.session_state.messages.append({
                            "role":       "assistant",
                            "content":    answer,
                            "nodes":      nodes,
                            "collection": coll_label,
                        })
                        st.session_state.query_count += 1

                    else:
                        err = response.error_message or "Unknown error"
                        st.error(f"Retrieval failed: {err}")
                        st.session_state.messages.append({
                            "role":    "assistant",
                            "content": f"⚠️ {err}",
                            "nodes":   [],
                        })

                except Exception as e:
                    logger.exception("Query error")
                    st.error(f"Error: {e}")

        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Embedded PDF viewer
# ══════════════════════════════════════════════════════════════════════════════
with col_pdf:
    st.markdown("### 📄 Source document")

    if st.session_state.pdf_url:
        fname = st.session_state.pdf_filename or ""
        page  = st.session_state.pdf_page or 1

        # Header bar
        st.markdown(f"""
        <div class="pdf-header">
            <span class="pdf-header-title">PDF·VIEWER</span>
            <span class="pdf-header-file">{fname}</span>
            <span class="pdf-header-page">Page {page}</span>
        </div>
        """, unsafe_allow_html=True)

        # Page jump — lightweight number input
        new_page = st.number_input(
            "Jump to page",
            min_value=1,
            value=page,
            step=1,
            label_visibility="collapsed",
            key=f"page_jump_{fname}_{page}",
        )
        if new_page != page:
            st.session_state.pdf_page = new_page
            st.session_state.pdf_url  = get_viewer_url(fname, new_page)
            st.rerun()

        # Embed iframe — fills the column
        viewer_url = st.session_state.pdf_url
        st.components.v1.iframe(
            src=viewer_url,
            height=680,
            scrolling=False,
        )

        # Open-in-new-tab link
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" '
            f'style="font-family:JetBrains Mono,monospace;font-size:11px;'
            f'color:#4f8ef7;text-decoration:none;">↗ Open in new tab</a>',
            unsafe_allow_html=True,
        )

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📄</div>
            <div>Ask a question to see the source PDF here</div>
        </div>
        """, unsafe_allow_html=True)