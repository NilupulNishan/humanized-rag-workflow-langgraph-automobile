"""
Streamlit RAG interface — chat UI + embedded PDF viewer

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from streamlit.components.v1 import html as st_html

from src.storage_manager import StorageManager
from src.retriever import SmartRetriever, MultiCollectionRetriever
from src.metadata_manager import MetadataManager
from pdf_server import get_viewer_url, SERVER_PORT, start_server_background


# ─── Path setup ───────────────────────────────────────────────────────────────
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

# ─── Start pdf_server ─────────────────────────────────────────────────────────


@st.cache_resource
def _boot_pdf_server():
    start_server_background()


_boot_pdf_server()

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
PDF_HTTP_BASE = "http://localhost:8000"


# ─── Session state defaults ───────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "selected_collection": None,
    "pdf_filename": None,
    "pdf_page": 1,
    "query_count": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def pdf_exists_on_disk(filename: str) -> bool:
    return bool(filename) and (PDF_DIR / filename).exists()


def get_pdf_http_url(filename: str, page: int) -> str:
    return f"{PDF_HTTP_BASE}/{quote(filename)}#page={int(page)}"


def render_pdf_viewer_pdfjs(filename: str, page: int, height: int = 720) -> None:
    """
    Renders PDF by injecting bytes directly as a JS Uint8Array literal.
    (No fetch(), no atob(), no cross-origin dependency.)
    """
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        st.warning(f"PDF not found: `{filename}` (expected under `{PDF_DIR}`)")
        return

    raw_bytes = pdf_path.read_bytes()
    hex_str = raw_bytes.hex()

    v_bg = "#f6f8ff"
    v_bar = "#ffffff"
    v_text = "#0b1b2b"
    v_border = "#cfe0ff"
    v_btn = "#f1f5ff"
    v_btn_h = "#e6eeff"
    v_accent = "#2563eb"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{
      height:100vh; overflow:hidden; background:{v_bg};
      font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
      display:flex; flex-direction:column;
    }}
    .bar {{
      background:{v_bar}; border-bottom:1px solid {v_border};
      padding:7px 12px; display:flex; align-items:center; gap:10px; flex:0 0 auto;
    }}
    .logo {{ font-size:10px; font-weight:700; letter-spacing:.12em; color:{v_text}; flex-shrink:0; }}
    .fname {{ flex:1; font-size:11px; color:{v_text}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .nav {{ display:flex; align-items:center; gap:6px; flex-shrink:0; }}
    .nav input {{
      width:60px; padding:2px 6px; font-size:11px;
      border:1px solid {v_border}; border-radius:4px; outline:none;
      background:transparent; color:{v_text}; text-align:center; font-family:inherit;
    }}
    .nav input:focus {{ border-color:{v_accent}; }}
    .total {{ font-size:10px; color:{v_text}; opacity:.55; white-space:nowrap; }}
    button {{
      padding:2px 10px; font-size:10px; font-weight:700;
      border:1px solid {v_border}; border-radius:4px;
      background:{v_btn}; color:{v_text}; cursor:pointer; font-family:inherit;
      transition:background .12s,border-color .12s;
    }}
    button:hover {{ background:{v_btn_h}; border-color:{v_accent}; }}
    .wrap {{
      flex:1 1 auto; overflow:auto; padding:12px;
      position:relative; display:flex; flex-direction:column; align-items:center;
    }}
    canvas {{
      display:none; background:white;
      border:1px solid {v_border}; border-radius:8px; max-width:100%;
      box-shadow:0 2px 12px rgba(37,99,235,.07);
    }}
    .loader {{
      display:flex; flex-direction:column; align-items:center;
      justify-content:center; gap:10px; width:100%; flex:1;
      color:{v_text}; font-size:11px; letter-spacing:.05em; opacity:.7;
    }}
    .spinner {{
      width:24px; height:24px;
      border:2px solid {v_border}; border-top-color:{v_accent};
      border-radius:50%; animation:spin .7s linear infinite;
    }}
    @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
    .err {{
      margin:12px; padding:12px 14px; color:#b91c1c;
      background:#fff1f1; border:1px solid #fca5a5;
      border-radius:6px; font-size:11px; line-height:1.6; width:100%;
      white-space:pre-wrap;
    }}
  </style>
</head>
<body>
  <div class="bar">
    <div class="logo">PDF·RAG</div>
    <div class="fname" title="{filename}">{filename}</div>
    <div class="nav">
      <input id="pg" type="number" min="1" value="{int(page)}"/>
      <span class="total" id="total"></span>
      <button id="go">Go</button>
    </div>
  </div>
  <div class="wrap" id="wrap">
    <div class="loader" id="loader">
      <div class="spinner"></div>
      <div id="loadmsg">Loading PDF.js…</div>
    </div>
    <canvas id="cv"></canvas>
    <div id="err" class="err" style="display:none;"></div>
  </div>

<script>
var HEX = "{hex_str}";
var START_PAGE = {int(page)};

function showErr(msg) {{
  document.getElementById('loader').style.display = 'none';
  document.getElementById('cv').style.display     = 'none';
  var el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent   = '⚠ ' + msg;
}}

function hexToUint8(hex) {{
  var len   = hex.length / 2;
  var bytes = new Uint8Array(len);
  for (var i = 0; i < len; i++) {{
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }}
  return bytes;
}}

function loadScript(url, cb, errCb) {{
  var s    = document.createElement('script');
  s.src    = url;
  s.onload = cb;
  s.onerror = errCb;
  document.head.appendChild(s);
}}

function startViewer() {{
  document.getElementById('loadmsg').textContent = 'Decoding PDF…';

  var pdfBytes;
  try {{
    pdfBytes = hexToUint8(HEX);
  }} catch(e) {{
    showErr('Hex decode failed: ' + e.message);
    return;
  }}

  // Use a blob URL for the worker to avoid CORS issues
  var workerCode = 'importScripts("https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js");';
  var workerBlob = new Blob([workerCode], {{ type: 'application/javascript' }});
  var workerUrl  = URL.createObjectURL(workerBlob);

  pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

  document.getElementById('loadmsg').textContent = 'Rendering…';

  var loadTask = pdfjsLib.getDocument({{ data: pdfBytes }});

  loadTask.promise.then(function(pdf) {{
    var pgInput = document.getElementById('pg');
    var totalEl = document.getElementById('total');
    var goBtn   = document.getElementById('go');
    var loader  = document.getElementById('loader');
    var canvas  = document.getElementById('cv');

    pgInput.max         = pdf.numPages;
    totalEl.textContent = '/ ' + pdf.numPages;

    var p = Math.min(Math.max(START_PAGE, 1), pdf.numPages);
    pgInput.value = p;

    function renderPage(num) {{
      loader.style.display = 'flex';
      canvas.style.display = 'none';
      document.getElementById('loadmsg').textContent = 'Rendering page ' + num + '…';

      pdf.getPage(num).then(function(page) {{
        var wrap       = document.getElementById('wrap');
        var containerW = wrap.clientWidth - 24;
        var vp1        = page.getViewport({{ scale: 1 }});
        var scale      = Math.min(2.5, Math.max(1.0, containerW / vp1.width));
        var vp         = page.getViewport({{ scale: scale }});

        canvas.width  = Math.floor(vp.width);
        canvas.height = Math.floor(vp.height);

        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        page.render({{ canvasContext: ctx, viewport: vp }}).promise.then(function() {{
          loader.style.display = 'none';
          canvas.style.display = 'block';
        }}).catch(function(e) {{
          showErr('Render error: ' + e.message);
        }});
      }}).catch(function(e) {{
        showErr('Page load error: ' + e.message);
      }});
    }}

    renderPage(p);

    goBtn.addEventListener('click', function() {{
      var v = parseInt(pgInput.value, 10);
      if (!isFinite(v) || v < 1) v = 1;
      if (v > pdf.numPages) v = pdf.numPages;
      pgInput.value = v;
      renderPage(v);
    }});

    pgInput.addEventListener('keydown', function(e) {{
      if (e.key === 'Enter') goBtn.click();
    }});

  }}).catch(function(e) {{
    showErr('PDF load error: ' + (e && e.message ? e.message : String(e)));
  }});
}}

loadScript(
  'https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.min.js',
  function() {{ startViewer(); }},
  function() {{
    loadScript(
      'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js',
      function() {{ startViewer(); }},
      function() {{ showErr('Could not load PDF.js from unpkg or jsdelivr.\\nCheck your internet connection or firewall.'); }}
    );
  }}
);
</script>
</body>
</html>
"""
    st_html(html, height=height, scrolling=False)


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
BG = "#f6f8ff"
SIDEBAR = "#ffffff"
PANEL = "#ffffff"
TEXT = "#0b1b2b"
BORDER = "#cfe0ff"
ACCENT = "#2563eb"
CHIP = "#f1f5ff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;500;600&display=swap');

:root {{
  --bg:{BG}; --sidebar:{SIDEBAR}; --panel:{PANEL};
  --text:{TEXT}; --border:{BORDER}; --accent:{ACCENT}; --chip:{CHIP};
}}

#MainMenu, footer {{ visibility:hidden; }}
.stApp {{ background-color:var(--bg); }}
.block-container {{ padding:1.4rem 2rem 2rem 2rem !important; background-color:var(--bg); }}
html,body,[class*="css"] {{ font-family:'Sora',sans-serif; color:var(--text); }}

[data-testid="stSidebar"] {{ background:var(--sidebar) !important; border-right:1px solid var(--border); }}
[data-testid="stSidebar"] * {{ color:var(--text) !important; }}
[data-testid="collapsedControl"] {{ display:block !important; visibility:visible !important; opacity:1 !important; z-index:9999 !important; }}
[data-testid="collapsedControl"] button {{ border:1px solid var(--border) !important; background:var(--sidebar) !important; }}
header[data-testid="stHeader"] {{ background:transparent !important; }}

.stButton > button {{
  background-color:var(--panel) !important; color:var(--text) !important;
  border:1px solid var(--border) !important;
  font-family:'JetBrains Mono',monospace !important; font-size:11px !important; transition:all 0.14s;
}}
.stButton > button:hover {{
  background-color:var(--accent) !important; color:#fff !important; border-color:var(--accent) !important;
}}

[data-testid="stChatMessage"] > div > div {{
  padding: 0px 10px 0px 10px !important;
}}
[data-testid="stChatMessage"] {{
  border-radius:10px; margin-bottom:6px; background-color:var(--panel);
  border:1px solid var(--border);
}}
[data-testid="stChatInput"] {{
  background-color:var(--panel);
  border:2px solid var(--border) !important;
  border-radius:8px;
}}
[data-testid="stChatInput"] input {{
  background-color:var(--panel);
  color:var(--text) !important;
}}
[data-testid="stChatInput"] input::placeholder {{
  color:var(--text) !important;
  opacity:0.6;
}}

.source-pill {{
  display:inline-flex; align-items:center; gap:6px;
  background:var(--chip); border:1px solid var(--border);
  border-radius:20px; padding:4px 12px 4px 10px;
  font-family:'JetBrains Mono',monospace; font-size:11px;
  color:var(--text); text-decoration:none;
  margin:3px 4px 0 0; cursor:pointer; transition:all .14s;
}}
.source-pill:hover {{ background:var(--panel); border-color:var(--accent); color:var(--accent); }}
.source-pill .dot {{ width:5px; height:5px; border-radius:50%; background:var(--accent); flex-shrink:0; }}

.coll-badge {{
  display:inline-block; background:var(--chip); border:1px solid var(--border);
  color:var(--text); font-family:'JetBrains Mono',monospace; font-size:10px;
  padding:2px 8px; border-radius:4px; letter-spacing:.06em;
  text-transform:uppercase; margin-bottom:6px;
}}

.stat-row {{ display:flex; gap:8px; margin-bottom:14px; }}
.stat-card {{ flex:1; background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:9px 10px; text-align:center; }}
.stat-num {{ font-family:'JetBrains Mono',monospace; font-size:19px; font-weight:600; color:var(--text); line-height:1; }}
.stat-label {{ font-size:9px; color:var(--text); letter-spacing:.06em; text-transform:uppercase; margin-top:3px; opacity:0.75; }}

.empty-pdf {{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  height:460px; gap:14px;
  color:var(--text); font-family:'JetBrains Mono',monospace; font-size:12px;
  letter-spacing:.05em; border:1px dashed var(--border); border-radius:12px; background:var(--panel);
}}
.empty-pdf .ei {{ font-size:40px; opacity:.5; }}

.pills {{
  margin-top: 10px;
  margin-bottom: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}

.source-pill,
.source-pill:visited,
.source-pill:hover,
.source-pill:active {{
  text-decoration: none !important;
}}
</style>
""", unsafe_allow_html=True)


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
      <div class="stat-card"><div class="stat-num">{len(collections)}</div><div class="stat-label">Collections</div></div>
      <div class="stat-card"><div class="stat-num">{total_chunks}</div><div class="stat-label">Chunks</div></div>
      <div class="stat-card"><div class="stat-num">{st.session_state.query_count}</div><div class="stat-label">Queries</div></div>
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
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text);margin-top:8px;opacity:.85;">
      ● pdf_server · port {SERVER_PORT}
    </div>
    """, unsafe_allow_html=True)
    st.caption("LlamaIndex · ChromaDB · Azure OpenAI")


# ─── Main columns ─────────────────────────────────────────────────────────────
col_chat, col_pdf = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Chat (fixed max-height + NO “messages under input” bug)
# ══════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown("### Ask a question")

    CHAT_HEIGHT = 650  # adjust 600–750
    chat_area = st.container(height=CHAT_HEIGHT)

    # Render history + create a placeholder at the bottom (INSIDE scroll area)
    with chat_area:
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
                        mm = MetadataManager()
                        pages = mm.extract_pages_from_nodes(nodes)
                        ranges = mm.merge_consecutive_pages(pages)
                        fname = mm.extract_filename_from_nodes(nodes)

                        pills = '<div class="pills">'
                        for start, end in ranges:
                            label = mm.format_page_range(start, end)
                            url = get_viewer_url(fname, start)
                            pills += (f'<a class="source-pill" href="{url}" target="_blank">'
                                      f'<span class="dot"></span>{label}</a>')
                        pills += "</div>"
                        st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])

        tail = st.empty()

    # Input OUTSIDE scroll area
    query = st.chat_input("Ask anything about your PDFs…")

    if query:
        # 1) save user message
        st.session_state.messages.append({"role": "user", "content": query})

        # 2) render user + assistant “Searching…” INSIDE chat_area via tail
        with tail.container():
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching…"):
                    try:
                        retriever = get_retriever(
                            st.session_state.selected_collection)
                        is_multi = isinstance(
                            retriever, MultiCollectionRetriever)
                        response = retriever.query_best(
                            query) if is_multi else retriever.query(query)
                        coll_label = response.collection_name if is_multi else st.session_state.selected_collection

                        if response.retrieval_successful:
                            answer = response.answer
                            nodes = response.source_nodes

                            if coll_label:
                                st.markdown(
                                    f'<span class="coll-badge">📁 {coll_label}</span>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown(answer)

                            mm = MetadataManager()
                            pages = mm.extract_pages_from_nodes(nodes)
                            ranges = mm.merge_consecutive_pages(pages)
                            fname = mm.extract_filename_from_nodes(nodes)

                            if pages and fname:
                                pills = '<div class="pills">'
                                for start, end in ranges:
                                    label = mm.format_page_range(start, end)
                                    url = get_viewer_url(fname, start)
                                    pills += (f'<a class="source-pill" href="{url}" target="_blank">'
                                              f'<span class="dot"></span>{label}</a>')
                                pills += "</div>"
                                st.markdown(pills, unsafe_allow_html=True)

                                if pdf_exists_on_disk(fname):
                                    st.session_state.pdf_filename = fname
                                    st.session_state.pdf_page = pages[0]
                                else:
                                    st.warning(
                                        f"Source PDF `{fname}` not found in `{PDF_DIR}`. "
                                        f"Available: {[f.name for f in PDF_DIR.glob('*.pdf')]}"
                                    )
                            elif pages and not fname:
                                st.warning(
                                    "Could not extract filename from source nodes.")

                            # store assistant message
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "nodes": nodes,
                                    "collection": coll_label,
                                }
                            )
                            st.session_state.query_count += 1

                        else:
                            err = response.error_message or "Unknown error"
                            st.error(f"Retrieval failed: {err}")
                            st.session_state.messages.append(
                                {"role": "assistant",
                                    "content": f"⚠️ {err}", "nodes": []}
                            )

                    except Exception as e:
                        logger.exception("Query error")
                        st.error(f"Error: {e}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"⚠️ {e}", "nodes": []}
                        )

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — PDF Viewer
# ══════════════════════════════════════════════════════════════════════════════
with col_pdf:
    st.markdown("### 📄 Source document")

    fname = st.session_state.pdf_filename
    page = int(st.session_state.pdf_page or 1)

    if fname:
        col_info, col_jump = st.columns([3, 1])

        with col_info:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;'
                f'color:var(--text);padding-top:6px;">'
                f'<span style="color:var(--accent);">●</span>&nbsp;{fname}'
                f'&nbsp;·&nbsp;page {page}</div>',
                unsafe_allow_html=True,
            )

        with col_jump:
            new_page = st.number_input(
                "page",
                min_value=1,
                value=page,
                step=1,
                label_visibility="collapsed",
                key=f"pjump_{fname}_{page}",
            )
            if int(new_page) != page:
                st.session_state.pdf_page = int(new_page)
                st.rerun()

        render_pdf_viewer_pdfjs(fname, page, height=720)

        viewer_url = get_viewer_url(fname, page)
        raw_url = get_pdf_http_url(fname, page)
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" style="font-family:JetBrains Mono,monospace;font-size:11px;'
            f'color:var(--text);text-decoration:none;border:1px solid var(--border);'
            f'padding:4px 12px;border-radius:4px;display:inline-block;margin-top:8px;">↗ open in new tab</a>'
            f'&nbsp;&nbsp;'
            f'<a href="{raw_url}" target="_blank" style="font-family:JetBrains Mono,monospace;font-size:11px;'
            f'color:var(--text);text-decoration:none;border:1px solid var(--border);'
            f'padding:4px 12px;border-radius:4px;display:inline-block;margin-top:8px;">↗ open raw PDF</a>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown("""
        <div class="empty-pdf">
          <div class="ei">📄</div>
          <div>Ask a question — the source PDF will appear here</div>
        </div>
        """, unsafe_allow_html=True)
