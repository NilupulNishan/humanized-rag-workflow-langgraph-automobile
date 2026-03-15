"""
Embeddings module for Azure OpenAI integration.
    Pass a custom httpx client with keep-alive and connection pooling.
    First call still pays TLS cost once. Every call after reuses the
    open socket. Cost drops from ~4s to ~0.5-0.8s per embed call.
"""
import httpx
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from config import settings


_http_client = httpx.Client(
    timeout=httpx.Timeout(
        connect=10.0,
        read=60.0,
        write=10.0,
        pool=5.0,
    ),
    limits=httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=60,
    ),
    http2=True,
)

# Async client for streaming (LlamaIndex streaming uses async internally)
_async_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0,),
    limits=httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=60,
    ),
    http2=True,
)

# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingsManager:
    """Manages LLM and embedding model initialization."""
    
    def __init__(self):
        """Initialize Azure OpenAI models."""
        settings.validate_config()
        
        # Initialize LLM
        self.llm = AzureOpenAI(
            model="gpt-4o",
            deployment_name=settings.AZURE_GPT4O_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.1,
            http_client=_http_client, # ← persistent connection
            async_http_client=_async_http_client,
        )
        
        # Initialize embedding model
        self.embed_model = AzureOpenAIEmbedding(
            model="text-embedding-3-large",
            deployment_name=settings.AZURE_EMBEDDING_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            http_client=_http_client,           # ← persistent connection
            async_http_client=_async_http_client,
        )
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        print("✓ Azure OpenAI models initialized")
    
    def get_llm(self):
        return self.llm
    
    def get_embed_model(self):
        return self.embed_model


if __name__ == "__main__":
    # Test embeddings manager
    manager = EmbeddingsManager()
    print(f"LLM Model: {manager.llm.model}")
    print(f"Embedding Model: {manager.embed_model.model}")
    print(f"Embedding Dimensions: {settings.EMBEDDING_DIMENSIONS}")