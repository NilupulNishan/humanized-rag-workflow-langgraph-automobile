"""
Production-grade configuration management.
"""
import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Model Deployments
AZURE_GPT4O_DEPLOYMENT = os.getenv("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")      
AZURE_GPT4O_MINI_DEPLOYMENT = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# Embedding Configuration
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 3072))

# Processing Configuration
CHUNK_SIZES_STR = os.getenv("CHUNK_SIZES", "4096,1024,512")
CHUNK_SIZES: List[int] = [int(size.strip()) for size in CHUNK_SIZES_STR.split(",")]
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 6))
ENABLE_AUTO_MERGING = os.getenv("ENABLE_AUTO_MERGING", "true").lower() == "true"

# Paths
PDF_DIRECTORY = PROJECT_ROOT / os.getenv("PDF_DIRECTORY", "data/pdfs")
CHROMA_DB_PATH = PROJECT_ROOT / os.getenv("CHROMA_DB_PATH", "data/chroma_db")
DOCSTORE_PATH = PROJECT_ROOT / os.getenv("DOCSTORE_PATH", "data/docstore")

# Page Link Configuration
PAGE_LINK_FORMAT = os.getenv("PAGE_LINK_FORMAT", "file://")  # or "http://" for web

# Create directories if they don't exist
PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)

# Langgraph 
REDIS_URL = os.getenv("REDIS_URL")
LANGGRAPH_PERSISTENCE = os.getenv("LANGGRAPH_PERSISTENCE")
CONFIDENCE_HIGH = os.getenv("CONFIDENCE_HIGH")
CONFIDENCE_LOW = os.getenv("CONFIDENCE_LOW")

def validate_config():
    """Validate that all required configuration is present."""
    required_vars = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    }
    
    missing_vars = [key for key, value in required_vars.items() if not value]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please check your .env file."
        )
    
    logger.info("✓ Configuration validated successfully")
    return True


def get_docstore_path(collection_name: str) -> Path:
    """Get path for a collection's docstore file."""
    return DOCSTORE_PATH / f"{collection_name}_docstore.json"


if __name__ == "__main__":
    # Test configuration
    validate_config()
    print(f"\nConfiguration:")
    print(f"  PDF Directory: {PDF_DIRECTORY}")
    print(f"  ChromaDB Path: {CHROMA_DB_PATH}")
    print(f"  Docstore Path: {DOCSTORE_PATH}")
    print(f"  Chunk Sizes: {CHUNK_SIZES}")
    print(f"  Embedding Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"  Auto-merging: {ENABLE_AUTO_MERGING}")
    print(f"  Log Level: {LOG_LEVEL}")