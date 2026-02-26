"""
Production-grade PDF loader with per-page document creation.
"""
import logging
from pathlib import Path
from typing import List, Tuple
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads PDFs with proper page-level metadata.
    
    Each page becomes a separate Document with complete metadata.
    LlamaIndex automatically propagates this metadata to all chunks.
    """
    
    def __init__(self):
        self.loader = PyMuPDFReader()
        logger.info("PDFLoader initialized")
    
    def get_collection_name(self, pdf_path: Path) -> str:
        """
        Generate a valid ChromaDB collection name from PDF filename.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Valid collection name (alphanumeric + underscores, lowercase)
        """
        name = pdf_path.stem
        # ChromaDB collection names: alphanumeric + underscores only
        collection_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return collection_name.lower()
    
    def load_pdf(self, pdf_path: Path) -> Tuple[List[Document], str]:
        """
        Load a PDF as separate page documents with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (list of page documents, collection_name)

        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        try:
            # Load PDF - PyMuPDF returns one Document per page
            page_docs = self.loader.load(file_path=pdf_path)
            
            if not page_docs:
                raise ValueError(f"No pages extracted from {pdf_path.name}")
            
            # Enhance each page document with complete metadata
            enhanced_docs = []
            for i, page_doc in enumerate(page_docs):
                page_num = i + 2  # 1-indexed page numbers
                
                # Create enhanced document with comprehensive metadata
                enhanced_doc = Document(
                    text=page_doc.get_content(),
                    metadata={
                        # File information
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path.absolute()),
                        'collection_name': self.get_collection_name(pdf_path),
                        
                        # Page information
                        'page': page_num,
                        'total_pages': len(page_docs),
                        
                        # Source tracking
                        'source': str(pdf_path),
                        'source_type': 'pdf',
                    }
                )
                enhanced_docs.append(enhanced_doc)
            
            collection_name = self.get_collection_name(pdf_path)
            
            logger.info(f"  ✓ Loaded {len(enhanced_docs)} pages from {pdf_path.name}")
            logger.debug(f"  Collection name: {collection_name}")
            
            return enhanced_docs, collection_name
            
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            raise
    
    def get_pdf_files(self, directory: Path) -> List[Path]:
        """
        Get all PDF files from a directory.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            Sorted list of PDF file paths
            
        Raises:
            ValueError: If directory doesn't exist or no PDFs found
        """
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory}")
        
        logger.info(f"Found {len(pdf_files)} PDF(s) in {directory}")
        
        return sorted(pdf_files)
    
    def load_all_pdfs(self, directory: Path) -> List[Tuple[List[Document], str, Path]]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of tuples: (page_documents, collection_name, pdf_path)
        """
        pdf_files = self.get_pdf_files(directory)
        results = []
        
        logger.info(f"\nLoading {len(pdf_files)} PDF(s)")
        print("=" * 80)
        
        for pdf_path in pdf_files:
            try:
                documents, collection_name = self.load_pdf(pdf_path)
                results.append((documents, collection_name, pdf_path))
                print(f"✓ {pdf_path.name} → {collection_name} ({len(documents)} pages)")
            except Exception as e:
                logger.error(f"Failed to load {pdf_path.name}: {e}")
                print(f"✗ {pdf_path.name} - Error: {e}")
                continue
        
        print("=" * 80)
        logger.info(f"Successfully loaded {len(results)}/{len(pdf_files)} PDFs")
        
        return results


if __name__ == "__main__":
    # Test the loader
    from config import settings
    
    loader = PDFLoader()
    
    try:
        results = loader.load_all_pdfs(settings.PDF_DIRECTORY)
        
        print(f"\nTest Results:")
        print(f"  Total PDFs loaded: {len(results)}")
        
        for docs, coll_name, path in results:
            print(f"\n  {path.name}:")
            print(f"    Collection: {coll_name}")
            print(f"    Pages: {len(docs)}")
            print(f"    First page metadata: {docs[0].metadata}")
            
    except Exception as e:
        print(f"Error: {e}")