"""
Smart retrieval with structured responses and metadata extraction.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from config import settings
from src.embeddings import EmbeddingsManager
from src.storage_manager import StorageManager
from src.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Structured query response with metadata."""
    answer: str
    source_nodes: List
    collection_name: str
    retrieval_successful: bool
    error_message: Optional[str] = None


class SmartRetriever:
    """
    Intelligent retrieval with auto-merging and metadata extraction.

    Features:
    - Automatic docstore loading
    - Auto-merging when available
    - Structured responses
    - Comprehensive metadata
    """

    def __init__(
        self,
        collection_name: str,
        verbose: bool = False,
        similarity_top_k: int = None
    ):
        """
        Initialize retriever for a collection.

        Args:
            collection_name: Name of collection to query
            verbose: Whether to show verbose logs
            similarity_top_k: Number of chunks to retrieve
        """
        self.collection_name = collection_name
        self.verbose = verbose
        self.similarity_top_k = similarity_top_k or settings.SIMILARITY_TOP_K

        # Initialize components
        self.embeddings_manager = EmbeddingsManager()
        self.storage_manager = StorageManager()

        # Load index
        try:
            self.index, self.storage_context, self.has_docstore = \
                self.storage_manager.load_index(
                    collection_name,
                    self.embeddings_manager.get_embed_model(),
                    enable_auto_merging=settings.ENABLE_AUTO_MERGING
                )

            mode = "auto-merging" if self.has_docstore else "standard"
            logger.info(f"SmartRetriever initialized ({mode}) for: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    def query(self, query_text: str, similarity_top_k: int = None) -> QueryResponse:
        """
        Query the collection and return structured response.

        Args:
            query_text: The question or query
            similarity_top_k: Override default similarity_top_k

        Returns:
            QueryResponse with answer and metadata
        """
        k = similarity_top_k or self.similarity_top_k

        try:
            logger.info(f"Querying: {query_text[:100]}...")

            # Get prompt template from PromptManager
            qa_prompt = PromptManager.get_qa_prompt()

            if self.has_docstore and settings.ENABLE_AUTO_MERGING:
                # Use auto-merging retrieval
                base_retriever = self.index.as_retriever(similarity_top_k=k)
                retriever = AutoMergingRetriever(
                    base_retriever,
                    storage_context=self.storage_context,
                    verbose=self.verbose
                )
                query_engine = RetrieverQueryEngine.from_args(retriever)
                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt}
                )
            else:
                # Use standard retrieval
                query_engine = self.index.as_query_engine(
                    similarity_top_k=k,
                    verbose=self.verbose
                )
                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt}
                )

            # Execute query
            response = query_engine.query(query_text)

            # Extract source nodes
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []

            logger.info(f"  ✓ Retrieved {len(source_nodes)} source nodes")

            return QueryResponse(
                answer=str(response),
                source_nodes=source_nodes,
                collection_name=self.collection_name,
                retrieval_successful=True
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResponse(
                answer="",
                source_nodes=[],
                collection_name=self.collection_name,
                retrieval_successful=False,
                error_message=str(e)
            )


class MultiCollectionRetriever:
    """Retriever that searches across multiple collections."""

    def __init__(
        self,
        collection_names: List[str] = None,
        verbose: bool = False
    ):
        """
        Initialize multi-collection retriever.

        Args:
            collection_names: List of collections to search (None = all)
            verbose: Whether to show verbose logs
        """
        storage_manager = StorageManager()

        if collection_names is None:
            collection_names = storage_manager.list_collections()

        if not collection_names:
            raise ValueError("No collections available")

        self.retrievers = {}
        for name in collection_names:
            try:
                self.retrievers[name] = SmartRetriever(name, verbose=verbose)
            except Exception as e:
                logger.warning(f"Failed to load collection {name}: {e}")

        if not self.retrievers:
            raise ValueError("Failed to load any collections")

        logger.info(f"MultiCollectionRetriever initialized with {len(self.retrievers)} collections")

    def query_all(self, query_text: str) -> dict:
        """
        Query all collections.

        Args:
            query_text: The question or query

        Returns:
            Dictionary mapping collection names to QueryResponse objects
        """
        results = {}

        for name, retriever in self.retrievers.items():
            results[name] = retriever.query(query_text)

        return results

    def query_best(self, query_text: str) -> QueryResponse:
        """
        Query all collections and return the best result.

        Args:
            query_text: The question or query

        Returns:
            Best QueryResponse (longest answer heuristic)
        """
        results = self.query_all(query_text)

        # Filter successful results
        successful = {k: v for k, v in results.items() if v.retrieval_successful}

        if not successful:
            return list(results.values())[0]

        # Return longest answer (simple heuristic)
        best_collection = max(successful.keys(), key=lambda k: len(successful[k].answer))

        return successful[best_collection]


if __name__ == "__main__":
    storage_manager = StorageManager()
    collections = storage_manager.list_collections()

    print(f"Testing SmartRetriever:\n")
    print(f"Available collections: {collections}\n")

    if collections:
        retriever = SmartRetriever(collections[0], verbose=True)
        response = retriever.query("What is this document about?")

        print(f"Response:")
        print(f"  Answer length: {len(response.answer)} chars")
        print(f"  Source nodes: {len(response.source_nodes)}")
        print(f"  Successful: {response.retrieval_successful}")
        print(f"  First 200 chars: {response.answer[:200]}...")
    else:
        print("No collections available. Run process_pdfs.py first.")