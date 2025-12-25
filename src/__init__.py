"""
Research Paper Question Answering - Source Package

This package contains all components for the RAG pipeline:
- pdf_processor: PDF text extraction
- chunker: Text chunking
- embedder: SPECTER embeddings
- vector_store: FAISS vector storage
- retriever: Two-stage retrieval
- generator: Qwen answer generation
- evaluator: Evaluation metrics
- pipeline: Main orchestration
"""

from src.pipeline import RAGPipeline, create_pipeline
from src.pdf_processor import PDFProcessor
from src.chunker import TextChunker, SemanticChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator
from src.evaluator import RAGEvaluator

__all__ = [
    "RAGPipeline",
    "create_pipeline",
    "PDFProcessor",
    "TextChunker",
    "SemanticChunker",
    "Embedder",
    "VectorStore",
    "Retriever",
    "Generator",
    "RAGEvaluator"
]

__version__ = "1.0.0"
