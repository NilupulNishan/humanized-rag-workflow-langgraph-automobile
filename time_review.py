"""
Drop-in replacement for retriever.py with per-step timing.
Usage: replace your retriever import temporarily to diagnose latency.
"""
import time
import logging
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

logger = logging.getLogger(__name__)


# ─── Timing helper ────────────────────────────────────────────────────────────

class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start
        color = "\033[91m" if self.elapsed > 3 else "\033[93m" if self.elapsed > 1 else "\033[92m"
        reset = "\033[0m"
        print(f"  {color}⏱  {self.label:<35} {self.elapsed:.3f}s{reset}")


# ─── Patched SmartRetriever ───────────────────────────────────────────────────

@dataclass
class QueryResponse:
    answer: str
    source_nodes: List
    collection_name: str
    retrieval_successful: bool
    error_message: Optional[str] = None
    timings: dict = None  # NEW: per-step timings


class TimedSmartRetriever:
    """
    Identical to SmartRetriever but prints per-step timing breakdown.
    Swap this in temporarily to find your bottleneck.
    """

    def __init__(self, collection_name: str, verbose: bool = False, similarity_top_k: int = None):
        from config import settings
        from src.embeddings import EmbeddingsManager
        from src.storage_manager import StorageManager

        self.collection_name = collection_name
        self.verbose = verbose
        self.similarity_top_k = similarity_top_k or settings.SIMILARITY_TOP_K
        self.settings = settings

        print(f"\n{'─'*55}")
        print(f"  Initializing TimedSmartRetriever: {collection_name}")
        print(f"{'─'*55}")

        with Timer("EmbeddingsManager init"):
            self.embeddings_manager = EmbeddingsManager()

        with Timer("StorageManager init"):
            self.storage_manager = StorageManager()

        with Timer("load_index (docstore JSON read)"):
            self.index, self.storage_context, self.has_docstore = \
                self.storage_manager.load_index(
                    collection_name,
                    self.embeddings_manager.get_embed_model(),
                    enable_auto_merging=settings.ENABLE_AUTO_MERGING
                )

        mode = "auto-merging" if self.has_docstore else "standard"
        print(f"  Mode: {mode}")
        print(f"{'─'*55}\n")

    def query(self, query_text: str, similarity_top_k: int = None) -> QueryResponse:
        k = similarity_top_k or self.similarity_top_k
        timings = {}

        print(f"\n{'─'*55}")
        print(f"  QUERY: {query_text[:60]}...")
        print(f"{'─'*55}")

        try:
            # Step 1: Build query engine
            with Timer("Build query engine") as t:
                if self.has_docstore and self.settings.ENABLE_AUTO_MERGING:
                    base_retriever = self.index.as_retriever(similarity_top_k=k)
                    retriever = AutoMergingRetriever(
                        base_retriever,
                        storage_context=self.storage_context,
                        verbose=self.verbose
                    )
                    query_engine = RetrieverQueryEngine.from_args(retriever)
                else:
                    query_engine = self.index.as_query_engine(
                        similarity_top_k=k,
                        verbose=self.verbose
                    )
            timings['build_engine'] = t.elapsed

            # Step 2: Embed query (hidden inside query_engine.query)
            # We split it out manually to time it separately
            with Timer("Embed query text (Azure API call)") as t:
                embed_model = self.embeddings_manager.get_embed_model()
                _ = embed_model.get_text_embedding(query_text)
            timings['embed_query'] = t.elapsed

            # Step 3: Full query (retrieval + LLM generation)
            # We time retrieval and LLM separately
            with Timer("ChromaDB vector search") as t:
                retriever_only = self.index.as_retriever(similarity_top_k=k)
                raw_nodes = retriever_only.retrieve(query_text)
            timings['vector_search'] = t.elapsed

            print(f"    → Retrieved {len(raw_nodes)} nodes from ChromaDB")

            # Step 4: AutoMerge (docstore lookup)
            if self.has_docstore and self.settings.ENABLE_AUTO_MERGING:
                with Timer("AutoMerging (docstore lookups)") as t:
                    from llama_index.core.retrievers import AutoMergingRetriever as AMR
                    am = AMR(
                        self.index.as_retriever(similarity_top_k=k),
                        storage_context=self.storage_context,
                        verbose=False
                    )
                    merged_nodes = am.retrieve(query_text)
                timings['auto_merge'] = t.elapsed
                print(f"    → After merge: {len(merged_nodes)} nodes")
            else:
                merged_nodes = raw_nodes

            # Step 5: LLM synthesis (the big one)
            with Timer("LLM answer synthesis (Azure API call)") as t:
                response = query_engine.query(query_text)
            timings['llm_synthesis'] = t.elapsed

            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []

            # Summary
            total = sum(timings.values())
            print(f"{'─'*55}")
            print(f"  \033[96mTOTAL: {total:.3f}s\033[0m")
            print(f"{'─'*55}\n")

            return QueryResponse(
                answer=str(response),
                source_nodes=source_nodes,
                collection_name=self.collection_name,
                retrieval_successful=True,
                timings=timings
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResponse(
                answer="",
                source_nodes=[],
                collection_name=self.collection_name,
                retrieval_successful=False,
                error_message=str(e),
                timings=timings
            )


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.storage_manager import StorageManager

    storage_manager = StorageManager()
    collections = storage_manager.list_collections()

    if not collections:
        print("No collections found. Run process_pdfs.py first.")
        sys.exit(1)

    collection = collections[0]
    print(f"Testing against collection: {collection}")

    retriever = TimedSmartRetriever(collection, verbose=False)

    query = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"

    response = retriever.query(query)

    print("Answer preview:")
    print(response.answer[:300])
    print("\nTimings breakdown:")
    for step, t in (response.timings or {}).items():
        print(f"  {step:<40} {t:.3f}s")