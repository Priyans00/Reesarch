"""
RAG Pipeline: Main orchestration for the Research Paper QA system.
Combines all components into a unified pipeline.
"""
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from config import (
    UPLOADS_DIR, 
    EMBEDDING_DIR,
    TOP_K_RERANK,
    DEVICE
)
from src.pdf_processor import PDFProcessor, ProcessedDocument
from src.chunker import SemanticChunker, TextChunk
from src.embedder import Embedder, BatchEmbedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator
from src.evaluator import RAGEvaluator, EvaluationResult


@dataclass
class QueryResult:
    """Result of a query to the RAG system."""
    question: str
    answer: str
    sources: List[Dict]
    context: str
    confidence: float
    abstained: bool
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "abstained": self.abstained
        }


class RAGPipeline:
    """
    Main RAG Pipeline for Research Paper Question Answering.
    
    Pipeline stages:
    1. Document ingestion (PDF processing)
    2. Text chunking
    3. Embedding generation
    4. Indexing in vector store
    5. Retrieval and reranking
    6. Answer generation
    """
    
    def __init__(
        self,
        auto_load: bool = True,
        device: str = DEVICE
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            auto_load: Whether to automatically load existing index
            device: Device to use for models
        """
        self.device = device
        
        # Initialize components
        print("Initializing RAG Pipeline...")
        
        # PDF Processor
        self.pdf_processor = PDFProcessor()
        
        # Chunker
        self.chunker = SemanticChunker()
        
        # Embedder (uses allenai-specter)
        self.embedder = Embedder(device=device)
        
        # Vector Store
        self.vector_store = VectorStore()
        
        # Retriever (with MS-MARCO reranking)
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            device=device
        )
        
        # Generator (uses Qwen2.5-0.5B)
        self.generator = Generator(device=device)
        
        # Evaluator
        self.evaluator = RAGEvaluator()
        
        # Document metadata storage
        self.documents: Dict[str, ProcessedDocument] = {}
        
        # Try to load existing index
        if auto_load:
            self._try_load_index()
        
        print("✓ Pipeline initialized")
    
    def _try_load_index(self):
        """Try to load existing vector store index."""
        if self.vector_store.load():
            print(f"  ✓ Loaded existing index with {self.vector_store.size} vectors")
        else:
            print("  ℹ No existing index found")
    
    # =========== Document Ingestion ===========
    
    def ingest_pdf(self, pdf_path: str) -> ProcessedDocument:
        """
        Ingest a single PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument object
        """
        print(f"\nIngesting: {pdf_path}")
        
        # Process PDF
        doc = self.pdf_processor.process_pdf(pdf_path)
        self.documents[doc.metadata.doc_id] = doc
        
        print(f"  ✓ Extracted {len(doc.full_text)} characters")
        print(f"  ✓ Title: {doc.metadata.title[:50]}...")
        
        return doc
    
    def ingest_directory(self, directory: str = None) -> List[ProcessedDocument]:
        """
        Ingest all PDFs from a directory.
        
        Args:
            directory: Directory path (defaults to UPLOADS_DIR)
            
        Returns:
            List of ProcessedDocument objects
        """
        if directory is None:
            directory = str(UPLOADS_DIR)
        
        documents = self.pdf_processor.process_directory(directory)
        
        for doc in documents:
            self.documents[doc.metadata.doc_id] = doc
        
        return documents
    
    # =========== Indexing ===========
    
    def index_documents(
        self, 
        documents: List[ProcessedDocument] = None,
        batch_size: int = 32
    ):
        """
        Index documents into the vector store.
        
        Args:
            documents: Documents to index (uses all stored if None)
            batch_size: Batch size for embedding
        """
        if documents is None:
            documents = list(self.documents.values())
        
        if not documents:
            print("No documents to index")
            return
        
        print(f"\nIndexing {len(documents)} documents...")
        
        # Chunk documents
        print("  Chunking documents...")
        all_chunks = self.chunker.chunk_documents(documents)
        
        if not all_chunks:
            print("  No chunks created")
            return
        
        # Generate embeddings
        print("  Generating embeddings...")
        embeddings = self.embedder.embed_chunks(all_chunks, batch_size=batch_size)
        
        # Add to vector store
        print("  Adding to vector store...")
        self.vector_store.add_embeddings(all_chunks, embeddings)
        
        # Save index
        self.vector_store.save()
        
        print(f"  ✓ Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    
    def add_document(self, pdf_path: str):
        """
        Convenience method to ingest and index a single document.
        
        Args:
            pdf_path: Path to the PDF file
        """
        doc = self.ingest_pdf(pdf_path)
        self.index_documents([doc])
    
    # =========== Query ===========
    
    def query(
        self,
        question: str,
        top_k: int = TOP_K_RERANK,
        max_context_length: int = 3000,
        filter_doc_ids: Optional[List[str]] = None,
        use_reranking: bool = True
    ) -> QueryResult:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            max_context_length: Maximum context length
            filter_doc_ids: Optional document filter
            use_reranking: Whether to use reranking
            
        Returns:
            QueryResult object with answer and sources
        """
        print(f"[DEBUG] Query called. Vector store size: {self.vector_store.size}")
        print(f"[DEBUG] Documents in store: {self.vector_store.doc_ids}")
        
        if self.vector_store.size == 0:
            return QueryResult(
                question=question,
                answer="No documents have been indexed. Please upload and index documents first.",
                sources=[],
                context="",
                confidence=0.0,
                abstained=True
            )
        
        # Retrieve relevant chunks
        context, source_chunks = self.retriever.get_context_for_generation(
            query=question,
            top_k=top_k,
            max_context_length=max_context_length,
            filter_doc_ids=filter_doc_ids
        )
        
        print(f"[DEBUG] Retrieved context length: {len(context)} chars, {len(source_chunks)} chunks")
        
        if not context:
            return QueryResult(
                question=question,
                answer="No relevant information found in the indexed documents.",
                sources=[],
                context="",
                confidence=0.0,
                abstained=True
            )
        
        # Generate answer
        result = self.generator.generate_with_sources(
            question=question,
            context=context,
            source_chunks=source_chunks
        )
        
        # Check if model abstained
        abstained = self.generator.check_abstention(result["answer"])
        
        # Calculate confidence based on retrieval scores
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        if retrieval_results:
            avg_score = sum(score for _, score in retrieval_results) / len(retrieval_results)
            confidence = min(avg_score, 1.0)
        else:
            confidence = 0.0
        
        return QueryResult(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            context=context,
            confidence=confidence,
            abstained=abstained
        )
    
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[QueryResult]:
        """
        Process multiple questions.
        
        Args:
            questions: List of questions
            **kwargs: Additional arguments for query()
            
        Returns:
            List of QueryResult objects
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nProcessing question {i}/{len(questions)}: {question[:50]}...")
            result = self.query(question, **kwargs)
            results.append(result)
        
        return results
    
    # =========== Evaluation ===========
    
    def evaluate(
        self,
        test_queries: List[Dict],
        run_evaluation: bool = True
    ) -> Dict:
        """
        Evaluate the pipeline on test queries.
        
        Args:
            test_queries: List of dicts with 'question' and optionally 
                         'ground_truth', 'relevant_doc_ids'
            run_evaluation: Whether to run full evaluation
            
        Returns:
            Evaluation report dictionary
        """
        self.evaluator.reset()
        
        for test in test_queries:
            question = test["question"]
            result = self.query(question)
            
            # Get retrieved chunks for evaluation
            retrieval_results = self.retriever.retrieve(question)
            retrieved_chunks = [chunk for chunk, _ in retrieval_results]
            
            self.evaluator.evaluate_single(
                question=question,
                answer=result.answer,
                context=result.context,
                retrieved_chunks=retrieved_chunks,
                relevant_doc_ids=test.get("relevant_doc_ids"),
                ground_truth=test.get("ground_truth")
            )
        
        report = self.evaluator.generate_report()
        self.evaluator.print_summary(report)
        
        return report.to_dict()
    
    # =========== Utilities ===========
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed documents."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": self.vector_store.size,
            "document_ids": list(self.documents.keys()),
            "vector_store": self.vector_store.get_statistics()
        }
    
    def list_documents(self) -> List[Dict]:
        """List all indexed documents."""
        return [
            {
                "doc_id": doc.metadata.doc_id,
                "title": doc.metadata.title,
                "filename": doc.metadata.filename,
                "num_pages": doc.metadata.num_pages
            }
            for doc in self.documents.values()
        ]
    
    def clear_index(self):
        """Clear the vector store and document storage."""
        self.vector_store.clear()
        self.documents.clear()
        print("Index and documents cleared")
    
    def reload_models(self):
        """Force reload all models."""
        self.embedder.model = None
        self.retriever.cross_encoder = None
        self.generator.model = None
        print("Models will be reloaded on next use")


class PipelineBuilder:
    """
    Builder pattern for configuring the RAG pipeline.
    """
    
    def __init__(self):
        self._config = {
            "auto_load": True,
            "device": DEVICE
        }
    
    def with_device(self, device: str):
        self._config["device"] = device
        return self
    
    def without_auto_load(self):
        self._config["auto_load"] = False
        return self
    
    def build(self) -> RAGPipeline:
        return RAGPipeline(**self._config)


# Convenience function for quick setup
def create_pipeline(auto_load: bool = True) -> RAGPipeline:
    """
    Create a RAG pipeline with default settings.
    
    Args:
        auto_load: Whether to load existing index
        
    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(auto_load=auto_load)


if __name__ == "__main__":
    # Example usage
    print("Research Paper QA Pipeline")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Show statistics
    stats = pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    
    # Example query (if documents are indexed)
    if stats['total_chunks'] > 0:
        print("\nExample Query:")
        result = pipeline.query("What is the main contribution of this paper?")
        print(f"  Question: {result.question}")
        print(f"  Answer: {result.answer[:200]}...")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Sources: {len(result.sources)}")
    else:
        print("\nNo documents indexed. Add PDFs to data/uploads and run:")
        print("  pipeline.ingest_directory()")
        print("  pipeline.index_documents()")
