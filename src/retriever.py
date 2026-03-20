import logging
from typing import Dict, List

import chromadb
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class RetrieverManager:

    def __init__(self):
        # ── Embeddings — initialized once ──────────────────────────────────────
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            model="text-embedding-3-large",
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )

        self.persist_directory   = str(settings.CHROMA_DB_PATH)
        self._retrievers:         Dict[str, any] = {}
        self._collection_metadata: List[dict]    = []

        logger.info("RetrieverManager initialized")
        logger.info(f"ChromaDB path: {self.persist_directory}")

    def load_all(self) -> Dict[str, any]:

        client      = chromadb.PersistentClient(path=self.persist_directory)
        collections = client.list_collections()

        if not collections:
            raise ValueError(
                f"No collections found in ChromaDB at {self.persist_directory}. "
                f"Run embeddings.py first."
            )

        logger.info(f"Found {len(collections)} collection(s) in ChromaDB")

        for col in collections:
            collection_name = col.name

            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # Build retriever with settings-driven k
            self._retrievers[collection_name] = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.SIMILARITY_TOP_K},
            )

            # Store metadata for dropdown
            self._collection_metadata.append({
                "collection_name": collection_name,
                "display_name":    self._format_display_name(collection_name),
            })

            logger.info(f"  ✓ Loaded: {collection_name}")

        return self._retrievers

    def get_retrievers(self) -> Dict[str, any]:

        if not self._retrievers:
            self.load_all()
        return self._retrievers

    def get_retriever(self, collection_name: str):
    
        retrievers = self.get_retrievers()

        if collection_name not in retrievers:
            available = list(retrievers.keys())
            raise KeyError(
                f"Collection '{collection_name}' not found. "
                f"Available: {available}"
            )

        return retrievers[collection_name]

    def get_dropdown_options(self) -> List[dict]:
        """
        Get collection list formatted for frontend dropdown.

        Returns:
            [
                {
                    "collection_name": "baic_bj30e30_user_manual_en_uop3",
                    "display_name":    "BAIC BJ30E30"
                },
                ...
            ]
        """
        if not self._collection_metadata:
            self.load_all()
        return self._collection_metadata

    def get_collection_names(self) -> List[str]:
        """Get raw list of collection names."""
        return list(self.get_retrievers().keys())

    def _format_display_name(self, collection_name: str) -> str:
        """
        Convert collection name to human readable display name.

        Examples:
            "baic_bj30e30_user_manual_en_uop3" → "BAIC BJ30E30"
            "biac_x55_ii_user_manual_en_5nyo"  → "BIAC X55 II"
        """
        parts = collection_name.split("_")

        # Words to strip from display name
        skip_words = {"en", "user", "manual"}

        # Filter out skip words and short random suffixes at the end
        # A suffix is considered random if it's short AND contains digits
        cleaned = []
        for part in parts:
            if part.lower() in skip_words:
                continue
            # Skip trailing random hash-like parts e.g. "uop3", "5nyo"
            if len(part) <= 4 and any(c.isdigit() for c in part):
                continue
            cleaned.append(part)

        return " ".join(p.upper() for p in cleaned)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    manager = RetrieverManager()

    # List all collections
    options = manager.get_dropdown_options()
    print("\nAvailable manuals:")
    for opt in options:
        print(f"  {opt['display_name']:<30} → {opt['collection_name']}")

    # Test a retriever
    if options:
        collection_name = options[1]["collection_name"]
        retriever = manager.get_retriever(collection_name)

        print(f"\nTesting retriever: {collection_name}")
        docs = retriever.invoke("I want to replace wiper ")

        print(f"Retrieved {len(docs)} docs:")
        for doc in docs:
            print(f"  Page {doc.metadata['page']}: {doc.page_content[:500]}...")