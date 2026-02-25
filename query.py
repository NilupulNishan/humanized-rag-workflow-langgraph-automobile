"""
Query interface — clean terminal UI with integrated PDF viewer.

Usage:
    python query.py                          # interactive mode
    python query.py "your question"          # single query, all collections
    python query.py collection "question"    # single query, specific collection
"""

import sys
import logging
import textwrap
from pathlib import Path

# ─── Path setup ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from colorama import init, Fore, Style
from src.retriever import SmartRetriever, MultiCollectionRetriever
from src.storage_manager import StorageManager
from src.source_formatter import SourceFormatter
from src.metadata_manager import MetadataManager
from pdf_server import start_server_background, get_viewer_url, open_pdf_at_page, SERVER_PORT

# ─── Init ─────────────────────────────────────────────────────────────────────
init()  # colorama

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Theme ────────────────────────────────────────────────────────────────────
DIM   = Style.DIM
RST   = Style.RESET_ALL
BOLD  = Style.BRIGHT

C_FRAME   = Fore.BLUE   + DIM
C_LABEL   = Fore.CYAN   + BOLD
C_ANSWER  = Fore.WHITE
C_SOURCE  = Fore.GREEN  + BOLD
C_LINK    = Fore.CYAN
C_PAGE    = Fore.YELLOW + BOLD
C_MUTED   = Fore.WHITE  + DIM
C_ERR     = Fore.RED
C_PROMPT  = Fore.BLUE   + BOLD
C_CMD     = Fore.MAGENTA + DIM
C_SUCCESS = Fore.GREEN  + BOLD

TERM_WIDTH = 80


# ─── Formatting helpers ───────────────────────────────────────────────────────

def rule(char="─", width=TERM_WIDTH, color=C_FRAME):
    return f"{color}{char * width}{RST}"


def header(title: str):
    print()
    print(rule("═"))
    pad = (TERM_WIDTH - len(title)) // 2
    print(f"{C_FRAME}║{RST}{' ' * pad}{C_LABEL}{title}{RST}{' ' * (TERM_WIDTH - pad - len(title) - 2)}{C_FRAME}║{RST}")
    print(rule("═"))
    print()


def section(label: str):
    """Print a subtle section divider."""
    label_str = f"  {label}  "
    bar_len = (TERM_WIDTH - len(label_str)) // 2
    print(f"\n{C_FRAME}{'─' * bar_len}{RST}{C_MUTED}{label_str}{RST}{C_FRAME}{'─' * bar_len}{RST}\n")


def print_answer(text: str):
    """Print the LLM answer with clean wrapping."""
    wrapper = textwrap.TextWrapper(width=TERM_WIDTH - 4, initial_indent="  ", subsequent_indent="  ")
    paragraphs = text.strip().split("\n")
    for para in paragraphs:
        if para.strip():
            print(f"{C_ANSWER}{wrapper.fill(para)}{RST}")
        else:
            print()


def print_sources(nodes: list, open_browser: bool = True):
    """
    Print source citations + clickable localhost links.
    Optionally opens the first source in the browser.
    """
    mm = MetadataManager()
    pages   = mm.extract_pages_from_nodes(nodes)
    ranges  = mm.merge_consecutive_pages(pages)
    fname   = mm.extract_filename_from_nodes(nodes)

    if not pages:
        print(f"\n  {C_MUTED}No source pages found.{RST}")
        return

    section("SOURCES")
    print(f"  {C_SOURCE}📄 {fname}{RST}\n")

    first_url = None
    first_page = None

    for i, (start, end) in enumerate(ranges):
        page_text = mm.format_page_range(start, end)
        viewer_url = get_viewer_url(fname, start)

        marker = "▸" if i == 0 else " "
        print(f"  {C_FRAME}{marker}{RST} {C_PAGE}{page_text:<14}{RST}  {C_LINK}{viewer_url}{RST}")

        if i == 0:
            first_url = viewer_url
            first_page = start

    print()

    if open_browser and first_url and first_page is not None:
        print(f"  {C_MUTED}Opening {fname} at {mm.format_page_range(first_page, first_page)} …{RST}")
        open_pdf_at_page(fname, first_page)

    print(f"\n  {C_MUTED}Tip: paste any URL above into your browser · [ / ] keys to flip pages{RST}")


def print_collection_list(collections: list):
    """Display available collections as a numbered list."""
    print(f"\n  {C_LABEL}AVAILABLE COLLECTIONS{RST}\n")
    print(f"  {C_CMD}  0  {RST}  {C_MUTED}Search ALL collections{RST}")
    for i, name in enumerate(collections, 1):
        print(f"  {C_CMD}{i:>3}{RST}  {name}")
    print()


def print_commands():
    """Print the command reference bar."""
    cmds = [
        ("quit / q", "exit"),
        ("change",   "switch collection"),
        ("open N",   "open source page N in browser"),
    ]
    line = "  " + "   ·   ".join(f"{C_CMD}{k}{RST} {C_MUTED}{v}{RST}" for k, v in cmds)
    print(line)
    print()


# ─── Collection selector ──────────────────────────────────────────────────────

def select_collection(collections: list) -> str | None:
    """Interactive collection picker. Returns name or None (= all)."""
    print_collection_list(collections)

    while True:
        try:
            raw = input(f"  {C_PROMPT}Select [{C_RST}0{C_PROMPT}–{len(collections)}]{RST}  ").strip()
            if not raw:
                continue
            n = int(raw)
            if n == 0:
                return None
            if 1 <= n <= len(collections):
                return collections[n - 1]
            print(f"  {C_ERR}Out of range.{RST}")
        except ValueError:
            print(f"  {C_ERR}Enter a number.{RST}")
        except KeyboardInterrupt:
            print()
            sys.exit(0)

# tiny hack so f-string above works
C_RST = RST


# ─── Retriever factory ────────────────────────────────────────────────────────

def build_retriever(selected: str | None, verbose=False):
    """Build either SmartRetriever or MultiCollectionRetriever."""
    if selected:
        return SmartRetriever(selected, verbose=verbose)
    return MultiCollectionRetriever(verbose=verbose)


# ─── Interactive mode ─────────────────────────────────────────────────────────

def interactive_query():
    header("PDF  QUERY  SYSTEM")

    # ── Check collections ────────────────────────────────────────────────────
    try:
        storage = StorageManager()
        collections = storage.list_collections()
    except Exception as e:
        print(f"\n  {C_ERR}Storage error: {e}{RST}")
        print(f"  {C_MUTED}Run process_pdfs.py first.{RST}\n")
        return 1

    if not collections:
        print(f"\n  {C_ERR}No collections found.{RST}")
        print(f"  {C_MUTED}Run process_pdfs.py first.{RST}\n")
        return 1

    # ── Start PDF server ─────────────────────────────────────────────────────
    started = start_server_background()
    if started:
        print(f"  {C_SUCCESS}✓{RST} {C_MUTED}PDF server  →  http://127.0.0.1:{SERVER_PORT}{RST}\n")

    # ── Pick collection ───────────────────────────────────────────────────────
    selected = select_collection(collections)
    label = selected if selected else "ALL collections"

    try:
        retriever = build_retriever(selected)
        print(f"\n  {C_SUCCESS}✓{RST} Connected to {C_LABEL}{label}{RST}\n")
    except Exception as e:
        print(f"\n  {C_ERR}Failed to load retriever: {e}{RST}\n")
        return 1

    # Keep last response's nodes so user can `open N`
    last_nodes: list = []

    print(rule())
    print_commands()

    # ── Query loop ────────────────────────────────────────────────────────────
    while True:
        try:
            raw = input(f"{C_PROMPT}▸ Query:{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {C_MUTED}Goodbye.{RST}\n")
            break

        if not raw:
            continue

        # Commands ────────────────────────────────────────────────────────────
        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            print(f"\n  {C_MUTED}Goodbye.{RST}\n")
            break

        if cmd == "change":
            selected = select_collection(collections)
            label = selected if selected else "ALL collections"
            try:
                retriever = build_retriever(selected)
                print(f"\n  {C_SUCCESS}✓{RST} Switched to {C_LABEL}{label}{RST}\n")
            except Exception as e:
                print(f"\n  {C_ERR}Error: {e}{RST}\n")
            continue

        # `open N` — open a specific source page
        if cmd.startswith("open"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit() and last_nodes:
                mm = MetadataManager()
                fname = mm.extract_filename_from_nodes(last_nodes)
                page  = int(parts[1])
                open_pdf_at_page(fname, page)
                print(f"  {C_MUTED}Opened page {page} of {fname}{RST}\n")
            else:
                print(f"  {C_MUTED}Usage: open <page_number>   (e.g. open 42){RST}\n")
            continue

        # Query ───────────────────────────────────────────────────────────────
        print(f"\n  {C_MUTED}Searching …{RST}")

        try:
            if isinstance(retriever, MultiCollectionRetriever):
                response = retriever.query_best(raw)
                if response.retrieval_successful:
                    print(f"\n  {C_MUTED}Best match: {C_LABEL}{response.collection_name}{RST}\n")
            else:
                response = retriever.query(raw)

            if response.retrieval_successful:
                section("ANSWER")
                print_answer(response.answer)
                last_nodes = response.source_nodes
                print_sources(last_nodes, open_browser=True)
            else:
                print(f"\n  {C_ERR}Error: {response.error_message}{RST}\n")

        except Exception as e:
            print(f"\n  {C_ERR}Query failed: {e}{RST}\n")
            logger.exception("Query error")

        print(rule())

    return 0


# ─── Single-shot (non-interactive) ────────────────────────────────────────────

def single_query(collection_name: str | None, query_text: str) -> int:
    """One-shot query for scripting / piping."""
    start_server_background()

    try:
        retriever = build_retriever(collection_name)

        if isinstance(retriever, MultiCollectionRetriever):
            response = retriever.query_best(query_text)
        else:
            response = retriever.query(query_text)

        if response.retrieval_successful:
            print(response.answer)
            # Plain source list
            mm = MetadataManager()
            pages  = mm.extract_pages_from_nodes(response.source_nodes)
            ranges = mm.merge_consecutive_pages(pages)
            fname  = mm.extract_filename_from_nodes(response.source_nodes)

            print(f"\nSources — {fname}")
            for start, end in ranges:
                print(f"  {mm.format_page_range(start, end)}: {get_viewer_url(fname, start)}")

            if pages:
                open_pdf_at_page(fname, pages[0])
            return 0
        else:
            print(f"Error: {response.error_message}", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    argc = len(sys.argv)
    if argc == 1:
        return interactive_query()
    elif argc == 2:
        return single_query(None, sys.argv[1])
    elif argc == 3:
        return single_query(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python query.py                           # interactive")
        print("  python query.py 'question'                # all collections")
        print("  python query.py collection_name 'question'")
        return 1


if __name__ == "__main__":
    sys.exit(main())