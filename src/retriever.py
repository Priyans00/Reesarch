"""
Retriever: Retrieves and reranks relevant document chunks.
Uses a two-stage approach:
1. Fast dense retrieval with SPECTER embeddings
2. Cross-encoder reranking with MS-MARCO MiniLM
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    SIMILARITY_THRESHOLD,
    DEVICE
)
from src.chunker import TextChunk
from src.embedder import Embedder
from src.vector_store import VectorStore


class Retriever:
    """
    Two-stage retriever with dense retrieval and cross-encoder reranking.
    
    Stage 1: Fast approximate retrieval using SPECTER embeddings
    Stage 2: Precise reranking using MS-MARCO cross-encoder
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker_model: str = RERANKER_MODEL,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        top_k_rerank: int = TOP_K_RERANK,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        device: str = DEVICE
    ):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance for query embedding
            vector_store: VectorStore instance for retrieval
            reranker_model: Cross-encoder model for reranking
            top_k_retrieval: Number of candidates for initial retrieval
            top_k_rerank: Number of results after reranking
            similarity_threshold: Minimum similarity score to include
            device: Device for cross-encoder
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker_model = reranker_model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        self.cross_encoder = None
    
    def load_reranker(self):
        """Load the cross-encoder reranking model."""
        if self.cross_encoder is None:
            print(f"Loading reranker: {self.reranker_model}")
            self.cross_encoder = CrossEncoder(
                self.reranker_model,
                max_length=512,
                device=self.device
            )
            print(f"  ✓ Reranker loaded on {self.device}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_doc_ids: Optional[List[str]] = None,
        use_reranking: bool = True
    ) -> List[Tuple[TextChunk, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to return (defaults to top_k_rerank)
            filter_doc_ids: Optional list of document IDs to filter
            use_reranking: Whether to use cross-encoder reranking
            
        Returns:
            List of (TextChunk, score) tuples
        """
        if top_k is None:
            top_k = self.top_k_rerank if use_reranking else self.top_k_retrieval
        
        # Stage 1: Dense retrieval
        retrieval_k = self.top_k_retrieval if use_reranking else top_k
        candidates = self._dense_retrieve(query, retrieval_k, filter_doc_ids)
        
        print(f"[DEBUG] Dense retrieval returned {len(candidates)} candidates")
        if candidates:
            print(f"[DEBUG] Dense scores range: {min(s for _, s in candidates):.4f} to {max(s for _, s in candidates):.4f}")
        
        if not candidates:
            return []
        
        # Stage 2: Reranking (if enabled)
        if use_reranking and len(candidates) > 1:
            candidates = self._rerank(query, candidates)
            print(f"[DEBUG] After reranking, scores range: {min(s for _, s in candidates):.4f} to {max(s for _, s in candidates):.4f}")
            # After reranking, just take top_k (no threshold - reranker already sorted by relevance)
            results = candidates[:top_k]
            print(f"[DEBUG] Returning top {len(results)} reranked results")
        else:
            # For dense-only retrieval, apply threshold
            results = [
                (chunk, score) 
                for chunk, score in candidates 
                if score >= self.similarity_threshold
            ][:top_k]
            print(f"[DEBUG] After threshold ({self.similarity_threshold}), {len(results)} results remain")
        
        return results
    
    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """
        Stage 1: Dense retrieval using embeddings.
        
        Args:
            query: User query
            top_k: Number of candidates to retrieve
            filter_doc_ids: Optional document filter
            
        Returns:
            List of (TextChunk, score) tuples
        """
        # Get query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_doc_ids=filter_doc_ids
        )
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[TextChunk, float]]
    ) -> List[Tuple[TextChunk, float]]:
        """
        Stage 2: Rerank candidates using cross-encoder.
        
        Args:
            query: User query
            candidates: List of (TextChunk, retrieval_score) tuples
            
        Returns:
            Reranked list of (TextChunk, rerank_score) tuples
        """
        self.load_reranker()
        
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        
        # Get reranking scores
        rerank_scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False
        )
        
        # Combine with chunks
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(candidates, rerank_scores)
        ]
        
        # Sort by reranking score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        filter_doc_ids: Optional[List[str]] = None,
        include_neighbors: bool = True
    ) -> List[Tuple[TextChunk, float]]:
        """
        Retrieve chunks with neighboring context.
        
        Args:
            query: User query
            top_k: Number of results
            filter_doc_ids: Document filter
            include_neighbors: Whether to include neighboring chunks
            
        Returns:
            List of (TextChunk, score) tuples with context
        """
        results = self.retrieve(query, top_k, filter_doc_ids)
        
        if not include_neighbors:
            return results
        
        # Expand with neighboring chunks
        expanded_results = []
        seen_chunk_ids = set()
        
        for chunk, score in results:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            
            # Add this chunk
            expanded_results.append((chunk, score))
            seen_chunk_ids.add(chunk.chunk_id)
            
            # Try to find neighbors from same document
            doc_chunks = self.vector_store.get_chunks_by_doc(chunk.doc_id)
            chunk_idx = chunk.metadata.get("chunk_index", -1)
            
            if chunk_idx >= 0:
                for doc_chunk in doc_chunks:
                    neighbor_idx = doc_chunk.metadata.get("chunk_index", -1)
                    # Include immediate neighbors
                    if abs(neighbor_idx - chunk_idx) == 1:
                        if doc_chunk.chunk_id not in seen_chunk_ids:
                            # Give neighbors a slightly lower score
                            expanded_results.append((doc_chunk, score * 0.8))
                            seen_chunk_ids.add(doc_chunk.chunk_id)
        
        # Re-sort by score
        expanded_results.sort(key=lambda x: x[1], reverse=True)
        
        return expanded_results
    
    def get_context_for_generation(
        self,
        query: str,
        top_k: int = None,
        max_context_length: int = 3000,
        filter_doc_ids: Optional[List[str]] = None
    ) -> Tuple[str, List[TextChunk]]:
        """
        Get formatted context for the generator.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            max_context_length: Maximum context length in characters
            filter_doc_ids: Document filter
            
        Returns:
            Tuple of (formatted_context, source_chunks)
        """
        if top_k is None:
            top_k = self.top_k_rerank
        
        results = self.retrieve(query, top_k, filter_doc_ids)
        
        if not results:
            return "", []
        
        # Format context with source markers
        context_parts = []
        source_chunks = []
        current_length = 0
        
        for i, (chunk, score) in enumerate(results, 1):
            # Format chunk with source marker
            source_info = f"[Source {i}]"
            if chunk.metadata.get("section"):
                source_info += f" [{chunk.metadata['section'].upper()}]"
            
            chunk_text = f"{source_info}\n{chunk.text}"
            
            # Check if adding this would exceed limit
            if current_length + len(chunk_text) > max_context_length:
                # Try to fit partial text
                remaining = max_context_length - current_length
                if remaining > 200:  # Only include if meaningful amount
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(chunk_text)
                    source_chunks.append(chunk)
                break
            
            context_parts.append(chunk_text)
            source_chunks.append(chunk)
            current_length += len(chunk_text)
        
        formatted_context = "\n\n".join(context_parts)
        
        return formatted_context, source_chunks


class HybridRetriever(Retriever):
    """
    Retriever with additional hybrid search capabilities.
    Combines dense and sparse retrieval for better coverage.
    """
    
    def __init__(self, *args, use_bm25: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bm25 = use_bm25
        self.bm25_index = None
    
    def _build_bm25_index(self):
        """Build BM25 index for sparse retrieval."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Get all chunk texts
            texts = [chunk.text for chunk in self.vector_store.chunks.values()]
            tokenized = [text.lower().split() for text in texts]
            
            self.bm25_index = BM25Okapi(tokenized)
            self.bm25_chunk_ids = list(self.vector_store.chunks.keys())
            
        except ImportError:
            print("Warning: rank_bm25 not installed. BM25 disabled.")
            self.use_bm25 = False
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_doc_ids: Optional[List[str]] = None,
        use_reranking: bool = True
    ) -> List[Tuple[TextChunk, float]]:
        """
        Hybrid retrieval combining dense and sparse methods.
        """
        if top_k is None:
            top_k = self.top_k_rerank if use_reranking else self.top_k_retrieval
        
        # Dense retrieval
        dense_results = self._dense_retrieve(
            query, 
            self.top_k_retrieval, 
            filter_doc_ids
        )
        
        # Combine with BM25 if enabled
        if self.use_bm25:
            bm25_results = self._bm25_retrieve(query, self.top_k_retrieval)
            dense_results = self._merge_results(dense_results, bm25_results)
        
        # Rerank
        if use_reranking and len(dense_results) > 1:
            dense_results = self._rerank(query, dense_results)
        
        # Apply threshold and limit
        results = [
            (chunk, score)
            for chunk, score in dense_results
            if score >= self.similarity_threshold
        ][:top_k]
        
        return results
    
    def _bm25_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[TextChunk, float]]:
        """Retrieve using BM25."""
        if self.bm25_index is None:
            self._build_bm25_index()
        
        if self.bm25_index is None:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.vector_store.chunks.get(chunk_id)
            if chunk:
                results.append((chunk, float(scores[idx])))
        
        return results
    
    def _merge_results(
        self,
        dense_results: List[Tuple[TextChunk, float]],
        sparse_results: List[Tuple[TextChunk, float]],
        dense_weight: float = 0.7
    ) -> List[Tuple[TextChunk, float]]:
        """Merge dense and sparse results using reciprocal rank fusion."""
        # Calculate RRF scores
        rrf_scores = {}
        k = 60  # RRF constant
        
        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0)
            rrf_scores[chunk.chunk_id] += dense_weight * (1 / (k + rank + 1))
        
        for rank, (chunk, _) in enumerate(sparse_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0)
            rrf_scores[chunk.chunk_id] += (1 - dense_weight) * (1 / (k + rank + 1))
        
        # Build result list
        chunk_map = {c.chunk_id: c for c, _ in dense_results + sparse_results}
        merged = [
            (chunk_map[cid], score)
            for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return merged


if __name__ == "__main__":
    # Test the retriever (requires embedder and vector store to be set up)
    print("Retriever module loaded successfully")
    print(f"Reranker model: {RERANKER_MODEL}")
    print(f"Top-K retrieval: {TOP_K_RETRIEVAL}")
    print(f"Top-K rerank: {TOP_K_RERANK}")
