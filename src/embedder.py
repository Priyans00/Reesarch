# Embedding generation module using SPECTER model for scientific papers

import os
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, DEVICE
from src.chunker import TextChunk

class Embedder:
    
    # Initializes the class with configuration parameters
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = EMBEDDING_DIMENSION
        
    # Loads the pre-trained model into memory
    def load_model(self):
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"  ✓ Model loaded on {self.device}")
            print(f"  ✓ Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    # Creates embedding vectors from text
    def embed_text(self, text: str) -> np.ndarray:
        self.load_model()
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        return embedding
    
    # Creates embeddings for multiple texts in batches
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        
        return embeddings
    
    # Creates embeddings for list of TextChunk objects
    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        self.load_model()
        
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, batch_size, show_progress)
        
        embedding_dict = {
            chunk_id: embedding 
            for chunk_id, embedding in zip(chunk_ids, embeddings)
        }
        
        return embedding_dict
    
    # Creates embedding for search query
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text(query)
    
    # Computes cosine similarity between query and document embeddings
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    # Finds and returns top-k most similar chunks to query
    def find_most_similar(
        self,
        query: str,
        chunks: List[TextChunk],
        chunk_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[tuple]:
        query_embedding = self.embed_query(query)
        
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = np.array([chunk_embeddings[cid] for cid in chunk_ids])
        
        similarities = self.compute_similarity(query_embedding, embeddings)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results

class BatchEmbedder(Embedder):
    
    # Initializes the class with configuration parameters
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    # Creates embeddings with optional caching for efficiency
    def embed_with_cache(
        self,
        texts: List[str],
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        embeddings = self.embed_texts(texts)
        
        if cache_key:
            self._cache[cache_key] = embeddings
        
        return embeddings
    
    # Clears all stored data
    def clear_cache(self):
        self._cache = {}
    
    # Creates embeddings for chunks with progress tracking
    def embed_chunks_progressive(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        save_every: int = 100
    ) -> Dict[str, np.ndarray]:
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
    embedder = Embedder()
    
    