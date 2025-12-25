"""
Embedder: Creates dense vector embeddings using allenai-specter.
SPECTER is specifically trained on scientific papers for semantic similarity.
"""
import os
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, DEVICE
from src.chunker import TextChunk


class Embedder:
    """
    Creates embeddings for text chunks using allenai-specter model.
    
    SPECTER (Scientific Paper Embeddings using Citation-informed Transformers)
    is trained on scientific papers and optimized for:
    - Document similarity
    - Citation prediction  
    - Scientific document retrieval
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = EMBEDDING_DIMENSION
        
    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"  ✓ Model loaded on {self.device}")
            print(f"  ✓ Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        self.load_model()
        
        # SPECTER works best with title + abstract format for papers
        # For chunks, we just use the text directly
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        return embedding
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Create embeddings for a list of text chunks.
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping chunk_id to embedding
        """
        self.load_model()
        
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, batch_size, show_progress)
        
        # Create mapping from chunk_id to embedding
        embedding_dict = {
            chunk_id: embedding 
            for chunk_id, embedding in zip(chunk_ids, embeddings)
        }
        
        return embedding_dict
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Create embedding for a search query.
        
        For SPECTER, queries are treated the same as documents.
        
        Args:
            query: Search query text
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        return self.embed_text(query)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Since embeddings are L2-normalized, dot product = cosine similarity.
        
        Args:
            query_embedding: Query embedding (embedding_dim,)
            document_embeddings: Document embeddings (num_docs, embedding_dim)
            
        Returns:
            Similarity scores (num_docs,)
        """
        # Ensure query is 2D for matrix multiplication
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity via dot product (embeddings are normalized)
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def find_most_similar(
        self,
        query: str,
        chunks: List[TextChunk],
        chunk_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find the most similar chunks to a query.
        
        Args:
            query: Search query
            chunks: List of TextChunk objects
            chunk_embeddings: Dictionary mapping chunk_id to embedding
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples, sorted by similarity
        """
        # Get query embedding
        query_embedding = self.embed_query(query)
        
        # Get embeddings in order
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = np.array([chunk_embeddings[cid] for cid in chunk_ids])
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with scores
        results = [
            (chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


class BatchEmbedder(Embedder):
    """
    Embedder with additional batch processing and caching capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def embed_with_cache(
        self,
        texts: List[str],
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Embed texts with optional caching.
        
        Args:
            texts: Texts to embed
            cache_key: Optional key for caching results
            
        Returns:
            Embeddings array
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        embeddings = self.embed_texts(texts)
        
        if cache_key:
            self._cache[cache_key] = embeddings
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}
    
    def embed_chunks_progressive(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        save_every: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Embed chunks with periodic saving for large collections.
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for encoding
            save_every: Save progress every N batches
            
        Returns:
            Dictionary mapping chunk_id to embedding
        """
        self.load_model()
        
        all_embeddings = {}
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch_chunks = chunks[i:i + batch_size]
            texts = [c.text for c in batch_chunks]
            
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            for chunk, emb in zip(batch_chunks, embeddings):
                all_embeddings[chunk.chunk_id] = emb
        
        return all_embeddings


if __name__ == "__main__":
    # Test the embedder
    embedder = Embedder()
    
    # Test single text embedding
    test_text = """
    We present a novel approach for question answering over scientific documents.
    Our method uses dense retrieval combined with a small language model to generate
    accurate answers grounded in the retrieved context.
    """
    
    embedding = embedder.embed_text(test_text)
    print(f"Single text embedding shape: {embedding.shape}")
    
    # Test batch embedding
    texts = [
        "Machine learning approaches for natural language processing.",
        "Deep neural networks have revolutionized computer vision tasks.",
        "Transformer models achieve state-of-the-art on many NLP benchmarks."
    ]
    
    embeddings = embedder.embed_texts(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query = "What are the best models for NLP?"
    query_emb = embedder.embed_query(query)
    
    similarities = embedder.compute_similarity(query_emb, embeddings)
    print(f"\nQuery: {query}")
    print("Similarities:")
    for text, sim in zip(texts, similarities):
        print(f"  {sim:.4f}: {text[:50]}...")
