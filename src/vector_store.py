"""
Vector Store: Efficient storage and retrieval of document embeddings.
Uses FAISS for fast similarity search.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import faiss

from config import (
    EMBEDDING_DIMENSION, 
    VECTOR_STORE_PATH, 
    METADATA_PATH,
    EMBEDDING_DIR
)
from src.chunker import TextChunk


class VectorStore:
    """
    Vector store for document embeddings using FAISS.
    
    Supports:
    - Adding embeddings with metadata
    - Fast similarity search
    - Persistence to disk
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIMENSION,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load the FAISS index
            metadata_path: Path to save/load metadata
        """
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path) if index_path else VECTOR_STORE_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        
        # Initialize FAISS index (using inner product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Metadata storage
        self.chunk_ids: List[str] = []
        self.chunks: Dict[str, TextChunk] = {}
        self.doc_ids: set = set()
        
    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.ntotal
    
    def add_embeddings(
        self,
        chunks: List[TextChunk],
        embeddings: Dict[str, np.ndarray]
    ):
        """
        Add embeddings to the vector store.
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Dictionary mapping chunk_id to embedding
        """
        if not chunks:
            return
        
        # Prepare vectors for FAISS
        vectors = []
        for chunk in chunks:
            if chunk.chunk_id in embeddings:
                vectors.append(embeddings[chunk.chunk_id])
                self.chunk_ids.append(chunk.chunk_id)
                self.chunks[chunk.chunk_id] = chunk
                self.doc_ids.add(chunk.doc_id)
        
        if vectors:
            vectors = np.array(vectors, dtype=np.float32)
            self.index.add(vectors)
            print(f"Added {len(vectors)} vectors to store. Total: {self.size}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_doc_ids: Optional list of doc_ids to filter results
            
        Returns:
            List of (TextChunk, score) tuples
        """
        if self.size == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Search more than needed if filtering
        search_k = min(top_k * 3 if filter_doc_ids else top_k, self.size)
        
        # Perform search
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Collect results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks.get(chunk_id)
            
            if chunk is None:
                continue
            
            # Apply filter if specified
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue
            
            results.append((chunk, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_text(
        self,
        query: str,
        embedder,
        top_k: int = 10,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """
        Search using text query (embedding is computed).
        
        Args:
            query: Text query
            embedder: Embedder instance for creating query embedding
            top_k: Number of results
            filter_doc_ids: Optional document filter
            
        Returns:
            List of (TextChunk, score) tuples
        """
        query_embedding = embedder.embed_query(query)
        return self.search(query_embedding, top_k, filter_doc_ids)
    
    def get_chunks_by_doc(self, doc_id: str) -> List[TextChunk]:
        """Get all chunks for a specific document."""
        return [
            chunk for chunk in self.chunks.values()
            if chunk.doc_id == doc_id
        ]
    
    def get_all_doc_ids(self) -> List[str]:
        """Get list of all document IDs in the store."""
        return list(self.doc_ids)
    
    def remove_document(self, doc_id: str):
        """
        Remove all chunks for a document.
        
        Note: FAISS doesn't support deletion, so we rebuild the index.
        """
        # Filter out chunks from the document
        chunks_to_keep = [
            (cid, self.chunks[cid])
            for cid in self.chunk_ids
            if self.chunks[cid].doc_id != doc_id
        ]
        
        if len(chunks_to_keep) == len(self.chunk_ids):
            return  # Document not found
        
        # Rebuild index
        self._rebuild_index(chunks_to_keep)
        self.doc_ids.discard(doc_id)
    
    def _rebuild_index(self, chunks_with_ids: List[Tuple[str, TextChunk]]):
        """Rebuild the FAISS index with filtered chunks."""
        # This requires re-embedding, so we store embeddings separately
        # For now, this is a limitation
        print("Warning: Rebuilding index requires re-embedding")
    
    def save(self):
        """Save the vector store to disk."""
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        metadata = {
            "chunk_ids": self.chunk_ids,
            "chunks": {cid: chunk.to_dict() for cid, chunk in self.chunks.items()},
            "doc_ids": list(self.doc_ids),
            "embedding_dim": self.embedding_dim
        }
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved vector store: {self.size} vectors")
        print(f"  Index: {self.index_path}")
        print(f"  Metadata: {self.metadata_path}")
    
    def load(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            print("No saved vector store found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.chunk_ids = metadata["chunk_ids"]
            self.chunks = {
                cid: TextChunk.from_dict(data) 
                for cid, data in metadata["chunks"].items()
            }
            self.doc_ids = set(metadata["doc_ids"])
            self.embedding_dim = metadata.get("embedding_dim", EMBEDDING_DIMENSION)
            
            print(f"Loaded vector store: {self.size} vectors from {len(self.doc_ids)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear all data from the vector store."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunk_ids = []
        self.chunks = {}
        self.doc_ids = set()
        print("Vector store cleared")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.size,
            "total_documents": len(self.doc_ids),
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dim,
            "index_type": "FlatIP (Inner Product)"
        }


class HybridVectorStore(VectorStore):
    """
    Vector store with IVF indexing for larger collections.
    Uses clustering for faster approximate search.
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIMENSION,
        n_clusters: int = 100,
        n_probe: int = 10,
        **kwargs
    ):
        """
        Initialize hybrid store with IVF index.
        
        Args:
            embedding_dim: Dimension of embeddings
            n_clusters: Number of clusters for IVF
            n_probe: Number of clusters to search
        """
        super().__init__(embedding_dim, **kwargs)
        
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self._is_trained = False
        self._training_vectors = []
    
    def _create_ivf_index(self, vectors: np.ndarray):
        """Create and train IVF index."""
        n_vectors = len(vectors)
        
        # Adjust clusters based on data size
        n_clusters = min(self.n_clusters, n_vectors // 10)
        n_clusters = max(n_clusters, 1)
        
        if n_vectors < 1000:
            # Use flat index for small collections
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Create IVF index
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                n_clusters,
                faiss.METRIC_INNER_PRODUCT
            )
            
            # Train the index
            self.index.train(vectors.astype(np.float32))
            self.index.nprobe = self.n_probe
        
        self._is_trained = True
    
    def add_embeddings(
        self,
        chunks: List[TextChunk],
        embeddings: Dict[str, np.ndarray]
    ):
        """Add embeddings, training index if needed."""
        if not chunks:
            return
        
        # Collect vectors
        vectors = []
        new_chunk_ids = []
        
        for chunk in chunks:
            if chunk.chunk_id in embeddings:
                vectors.append(embeddings[chunk.chunk_id])
                new_chunk_ids.append(chunk.chunk_id)
                self.chunks[chunk.chunk_id] = chunk
                self.doc_ids.add(chunk.doc_id)
        
        if not vectors:
            return
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # Train index if not already trained
        if not self._is_trained:
            self._create_ivf_index(vectors)
        
        # Add vectors
        self.index.add(vectors)
        self.chunk_ids.extend(new_chunk_ids)
        
        print(f"Added {len(vectors)} vectors. Total: {self.size}")


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    
    # Create test chunks
    test_chunks = [
        TextChunk(
            chunk_id="test_1",
            doc_id="doc_1",
            text="Machine learning for NLP",
            metadata={"section": "intro"}
        ),
        TextChunk(
            chunk_id="test_2",
            doc_id="doc_1",
            text="Deep learning architectures",
            metadata={"section": "methods"}
        ),
        TextChunk(
            chunk_id="test_3",
            doc_id="doc_2",
            text="Computer vision applications",
            metadata={"section": "intro"}
        ),
    ]
    
    # Create fake embeddings
    embeddings = {
        "test_1": np.random.randn(768).astype(np.float32),
        "test_2": np.random.randn(768).astype(np.float32),
        "test_3": np.random.randn(768).astype(np.float32),
    }
    
    # Normalize embeddings
    for k, v in embeddings.items():
        embeddings[k] = v / np.linalg.norm(v)
    
    # Add to store
    store.add_embeddings(test_chunks, embeddings)
    
    # Test search
    query_emb = np.random.randn(768).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    results = store.search(query_emb, top_k=2)
    print(f"\nSearch results:")
    for chunk, score in results:
        print(f"  {score:.4f}: {chunk.text}")
    
    # Test statistics
    print(f"\nStatistics: {store.get_statistics()}")
    
    # Test save/load
    store.save()
    
    new_store = VectorStore()
    new_store.load()
    print(f"\nLoaded store size: {new_store.size}")
