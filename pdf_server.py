"""
Local PDF server - serves PDFs from data/pdfs/ and opens them at exact pages.

Usage:
    # Start the server (runs in background)
    python pdf_server.py

    # Or import and use programmatically
    from pdf_server import open_pdf_at_page, start_server_background
"""

import os
import sys
import threading
import webbrowser
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
PDF_DIRECTORY = PROJECT_ROOT / "data" / "pdfs"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7654
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ─── HTML Viewer Template ─────────────────────────────────────────────────────

VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{filename} — PDF Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #0d0f12;
    --surface: #151820;
    --border: #1e2430;
    --accent: #4f8ef7;
    --accent-dim: #2a4a8a;
    --text: #e2e8f4;
    --text-muted: #6b7a99;
    --success: #34d399;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Sora', sans-serif;
  }}

  html, body {{
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    overflow: hidden;
  }}

  /* ── Top bar ── */
  .topbar {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    height: 54px;
    flex-shrink: 0;
  }}

  .logo {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    white-space: nowrap;
  }}

  .filename {{
    font-size: 13px;
    color: var(--text-muted);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .filename strong {{
    color: var(--text);
    font-weight: 600;
  }}

  .page-badge {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    font-family: var(--mono);
    font-size: 12px;
  }}

  .page-badge .dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
  }}

  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
  }}

  .nav-btn {{
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 6px;
    padding: 5px 14px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }}
  .nav-btn:hover {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }}

  /* ── Page jump input ── */
  .jump-group {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .jump-label {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }}
  .jump-input {{
    width: 56px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 4px 8px;
    text-align: center;
    outline: none;
    transition: border-color 0.15s;
  }}
  .jump-input:focus {{ border-color: var(--accent); }}

  /* ── PDF embed ── */
  .viewer-wrap {{
    height: calc(100vh - 54px);
    width: 100%;
  }}

  iframe {{
    width: 100%;
    height: 100%;
    border: none;
    display: block;
  }}

  /* ── Fade-in ── */
  body {{ animation: fadein 0.3s ease; }}
  @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
</style>
</head>
<body>

<div class="topbar">
  <span class="logo">PDF·RAG</span>
  <span class="filename"><strong>{filename}</strong></span>
  <div class="page-badge">
    <span class="dot"></span>
    <span id="page-label">Page {page}</span>
  </div>
  <div class="jump-group">
    <span class="jump-label">GO TO</span>
    <input class="jump-input" id="jump-input" type="number" min="1" value="{page}" placeholder="{page}">
    <button class="nav-btn" onclick="goToPage()">↵</button>
  </div>
  <button class="nav-btn" onclick="window.open('/pdf/{pdf_path}', '_blank')">↗ RAW</button>
</div>

<div class="viewer-wrap">
  <iframe id="pdf-frame" src="/pdf/{pdf_path}#page={page}" title="{filename}"></iframe>
</div>

<script>
  const pdfPath = '{pdf_path}';
  let currentPage = {page};

  function goToPage() {{
    const input = document.getElementById('jump-input');
    const page = parseInt(input.value, 10);
    if (!page || page < 1) return;
    currentPage = page;
    document.getElementById('page-label').textContent = 'Page ' + page;
    document.getElementById('pdf-frame').src = '/pdf/' + pdfPath + '#page=' + page;
  }}

  document.getElementById('jump-input').addEventListener('keydown', e => {{
    if (e.key === 'Enter') goToPage();
  }});

  // Keyboard shortcuts: [ and ] to go back/forward pages
  document.addEventListener('keydown', e => {{
    if (e.key === '[' && currentPage > 1) {{
      document.getElementById('jump-input').value = --currentPage;
      goToPage();
    }} else if (e.key === ']') {{
      document.getElementById('jump-input').value = ++currentPage;
      goToPage();
    }}
  }});
</script>
</body>
</html>
"""

# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class PDFHandler(BaseHTTPRequestHandler):
    """Handles PDF serving and viewer page requests."""

    def log_message(self, format, *args):
        """Suppress default HTTP logs."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ── GET /viewer?file=name.pdf&page=N ──────────────────────────────
        if path == "/viewer":
            params = parse_qs(parsed.query)
            filename = params.get("file", [""])[0]
            page = int(params.get("page", [1])[0])

            pdf_file = PDF_DIRECTORY / filename
            if not pdf_file.exists():
                self._send_404(f"PDF not found: {filename}")
                return

            html = VIEWER_HTML.format(
                filename=filename,
                page=page,
                pdf_path=filename,
            )
            self._send_html(html)

        # ── GET /pdf/<filename> ───────────────────────────────────────────
        elif path.startswith("/pdf/"):
            filename = unquote(path[5:])  # strip /pdf/
            pdf_file = PDF_DIRECTORY / filename

            if not pdf_file.exists():
                self._send_404(f"PDF not found: {filename}")
                return

            self._send_pdf(pdf_file)

        # ── GET /list — list available PDFs (health check) ────────────────
        elif path == "/list":
            pdfs = [f.name for f in PDF_DIRECTORY.glob("*.pdf")]
            body = "\n".join(pdfs).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── GET /health ───────────────────────────────────────────────────
        elif path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        else:
            self._send_404("Unknown route")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_pdf(self, pdf_path: Path):
        size = pdf_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        # Allow browser PDF viewer to navigate pages
        self.send_header("Content-Disposition", f'inline; filename="{pdf_path.name}"')
        self.end_headers()
        with open(pdf_path, "rb") as f:
            self.wfile.write(f.read())

    def _send_404(self, msg: str):
        body = msg.encode()
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)


# ─── Public API ───────────────────────────────────────────────────────────────

_server_instance: HTTPServer | None = None
_server_thread: threading.Thread | None = None


def start_server(host=SERVER_HOST, port=SERVER_PORT) -> bool:
    """
    Start the PDF server (blocking). Used when running as __main__.
    """
    global _server_instance
    _server_instance = HTTPServer((host, port), PDFHandler)
    print(f"  PDF server running at http://{host}:{port}")
    try:
        _server_instance.serve_forever()
    except KeyboardInterrupt:
        pass
    return True


def start_server_background(host=SERVER_HOST, port=SERVER_PORT) -> bool:
    """
    Start the PDF server in a background daemon thread.

    Returns True if started successfully, False if already running or failed.
    """
    global _server_instance, _server_thread

    if _server_thread and _server_thread.is_alive():
        return True  # already running

    try:
        _server_instance = HTTPServer((host, port), PDFHandler)
        _server_thread = threading.Thread(
            target=_server_instance.serve_forever,
            daemon=True,
            name="pdf-server",
        )
        _server_thread.start()
        logger.info(f"PDF server started at {SERVER_URL}")
        return True
    except OSError as e:
        if "Address already in use" in str(e):
            # Server is probably already running from a previous session
            logger.warning(f"Port {port} already in use — assuming server is running")
            return True
        logger.error(f"Failed to start PDF server: {e}")
        return False


def get_viewer_url(filename: str, page: int = 1) -> str:
    """
    Get the viewer URL for a PDF file at a specific page.

    Args:
        filename: PDF filename (just the name, not the full path)
        page: Page number (1-indexed)

    Returns:
        Full viewer URL string
    """
    return f"{SERVER_URL}/viewer?file={filename}&page={page}"


def open_pdf_at_page(filename: str, page: int = 1) -> bool:
    """
    Open a PDF in the browser at a specific page.

    Starts the server if not already running.

    Args:
        filename: PDF filename (just the name, not the full path)
        page: Page number (1-indexed)

    Returns:
        True if browser was opened successfully
    """
    # Ensure server is running
    start_server_background()

    url = get_viewer_url(filename, page)
    try:
        webbrowser.open(url)
        logger.info(f"Opened browser: {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        return False


def is_server_running() -> bool:
    """Check if the server thread is alive."""
    return _server_thread is not None and _server_thread.is_alive()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print(f"  PDF Server")
    print(f"{'═'*60}")
    print(f"  Serving PDFs from: {PDF_DIRECTORY}")
    print(f"  Press Ctrl+C to stop\n")

    if not PDF_DIRECTORY.exists():
        print(f"  ⚠  Directory not found: {PDF_DIRECTORY}")
        print(f"     Create it and add PDF files, then restart.\n")

    start_server()
