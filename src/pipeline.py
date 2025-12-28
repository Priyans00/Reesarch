# Main RAG pipeline orchestrating all components for question answering

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
    question: str
    answer: str
    sources: List[Dict]
    context: str
    confidence: float
    abstained: bool
    
    # Converts object to dictionary representation
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "abstained": self.abstained
        }

class RAGPipeline:
    
    # Initializes the class with configuration parameters
    def __init__(
        self,
        auto_load: bool = True,
        device: str = DEVICE
    ):
        self.device = device
        
        print("Initializing RAG Pipeline...")
        
        self.pdf_processor = PDFProcessor()
        
        self.chunker = SemanticChunker()
        
        self.embedder = Embedder(device=device)
        
        self.vector_store = VectorStore()
        
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            device=device
        )
        
        self.generator = Generator(device=device)
        
        self.evaluator = RAGEvaluator()
        
        self.documents: Dict[str, ProcessedDocument] = {}
        
        if auto_load:
            self._try_load_index()
        
        print("✓ Pipeline initialized")
    
    # Loads data from disk
    def _try_load_index(self):
        if self.vector_store.load():
            print(f"  ✓ Loaded existing index with {self.vector_store.size} vectors")
        else:
            print("  ℹ No existing index found")
    
    
    # Ingest Pdf
    def ingest_pdf(self, pdf_path: str) -> ProcessedDocument:
        print(f"\nIngesting: {pdf_path}")
        
        doc = self.pdf_processor.process_pdf(pdf_path)
        self.documents[doc.metadata.doc_id] = doc
        
        print(f"  ✓ Extracted {len(doc.full_text)} characters")
        print(f"  ✓ Title: {doc.metadata.title[:50]}...")
        
        return doc
    
    # Ingest Directory
    def ingest_directory(self, directory: str = None) -> List[ProcessedDocument]:
        if directory is None:
            directory = str(UPLOADS_DIR)
        
        documents = self.pdf_processor.process_directory(directory)
        
        for doc in documents:
            self.documents[doc.metadata.doc_id] = doc
        
        return documents
    
    
    # Index Documents
    def index_documents(
        self, 
        documents: List[ProcessedDocument] = None,
        batch_size: int = 32
    ):
        if documents is None:
            documents = list(self.documents.values())
        
        if not documents:
            print("No documents to index")
            return
        
        print(f"\nIndexing {len(documents)} documents...")
        
        print("  Chunking documents...")
        all_chunks = self.chunker.chunk_documents(documents)
        
        if not all_chunks:
            print("  No chunks created")
            return
        
        print("  Generating embeddings...")
        embeddings = self.embedder.embed_chunks(all_chunks, batch_size=batch_size)
        
        print("  Adding to vector store...")
        self.vector_store.add_embeddings(all_chunks, embeddings)
        
        self.vector_store.save()
        
        print(f"  ✓ Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Adds new items to the collection
    def add_document(self, pdf_path: str):
        doc = self.ingest_pdf(pdf_path)
        self.index_documents([doc])
    
    
    # Query
    def query(
        self,
        question: str,
        top_k: int = TOP_K_RERANK,
        max_context_length: int = 3000,
        filter_doc_ids: Optional[List[str]] = None,
        use_reranking: bool = True
    ) -> QueryResult:
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
        
        result = self.generator.generate_with_sources(
            question=question,
            context=context,
            source_chunks=source_chunks
        )
        
        abstained = self.generator.check_abstention(result["answer"])
        
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        if retrieval_results:
            avg_score = sum(score for _, score in retrieval_results) / len(retrieval_results)
            confidence = max(0.0, min(avg_score, 1.0))
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
    
    # Batch Query
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[QueryResult]:
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nProcessing question {i}/{len(questions)}: {question[:50]}...")
            result = self.query(question, **kwargs)
            results.append(result)
        
        return results
    
    
    # Evaluates performance using metrics
    def evaluate(
        self,
        test_queries: List[Dict],
        run_evaluation: bool = True
    ) -> Dict:
        self.evaluator.reset()
        
        for test in test_queries:
            question = test["question"]
            result = self.query(question)
            
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
    
    
    # Retrieves items from the collection
    def get_statistics(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "total_chunks": self.vector_store.size,
            "document_ids": list(self.documents.keys()),
            "vector_store": self.vector_store.get_statistics()
        }
    
    # List Documents
    def list_documents(self) -> List[Dict]:
        return [
            {
                "doc_id": doc.metadata.doc_id,
                "title": doc.metadata.title,
                "filename": doc.metadata.filename,
                "num_pages": doc.metadata.num_pages
            }
            for doc in self.documents.values()
        ]
    
    # Clears all stored data
    def clear_index(self):
        self.vector_store.clear()
        self.documents.clear()
        print("Index and documents cleared")
    
    # Loads the pre-trained model into memory
    def reload_models(self):
        self.embedder.model = None
        self.retriever.cross_encoder = None
        self.generator.model = None
        print("Models will be reloaded on next use")

class PipelineBuilder:
    
    # Initializes the class with configuration parameters
    def __init__(self):
        self._config = {
            "auto_load": True,
            "device": DEVICE
        }
    
    # With Device
    def with_device(self, device: str):
        self._config["device"] = device
        return self
    
    # Loads data from disk
    def without_auto_load(self):
        self._config["auto_load"] = False
        return self
    
    # Build
    def build(self) -> RAGPipeline:
        return RAGPipeline(**self._config)

# Create Pipeline
def create_pipeline(auto_load: bool = True) -> RAGPipeline:
    return RAGPipeline(auto_load=auto_load)

if __name__ == "__main__":
    print("Research Paper QA Pipeline")
    print("=" * 50)
    
    pipeline = create_pipeline()
    
    stats = pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    
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
