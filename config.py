"""
Configuration settings for the Research Paper QA System.
Uses small, efficient models optimized for each task.
"""
import os
from pathlib import Path

# ============== Paths ==============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embedding"

# Create directories if they don't exist
for dir_path in [UPLOADS_DIR, PROCESSED_DIR, EMBEDDING_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============== Model Configuration ==============
# Embedding Model: allenai-specter (trained on scientific papers)
EMBEDDING_MODEL = "sentence-transformers/allenai-specter"
EMBEDDING_DIMENSION = 768

# Reranking Model: Cross-encoder for MS-MARCO
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Generation Model: Qwen2.5-0.5B-Instruct (lightweight but capable)
GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATOR_MAX_NEW_TOKENS = 512
GENERATOR_TEMPERATURE = 0.3
GENERATOR_TOP_P = 0.9

# ============== Chunking Configuration ==============
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens
MIN_CHUNK_SIZE = 100  # minimum chunk size in characters

# ============== Retrieval Configuration ==============
TOP_K_RETRIEVAL = 10  # Number of chunks to retrieve initially
TOP_K_RERANK = 5  # Number of chunks after reranking
SIMILARITY_THRESHOLD = -5.0  # Minimum similarity score (cross-encoder outputs logits, typically -10 to +10)

# ============== Vector Store Configuration ==============
VECTOR_STORE_TYPE = "faiss"  # Using FAISS for efficient similarity search
VECTOR_STORE_PATH = EMBEDDING_DIR / "vector_store.index"
METADATA_PATH = EMBEDDING_DIR / "metadata.json"

# ============== Device Configuration ==============
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============== Prompt Templates ==============
SYSTEM_PROMPT = """You are a precise research assistant that answers questions based ONLY on the provided context from research papers. 

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in the context
2. If the context doesn't contain enough information, say "I cannot answer this based on the provided context"
3. Never make up or infer information not present in the context
4. Cite specific parts of the context when answering
5. Keep answers concise and factual"""

QA_PROMPT_TEMPLATE = """Context from research papers:
{context}

Question: {question}

Based strictly on the above context, provide a precise answer. If the information is not in the context, state that clearly."""

# ============== Evaluation Metrics ==============
EVAL_METRICS = ["precision", "recall", "f1", "mrr"]
