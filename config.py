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
SYSTEM_PROMPT = """You are a concise, factual research assistant. Follow these rules exactly:
- Use ONLY the provided context; do not hallucinate or infer.
- If the context lacks necessary information, set "abstain": true and provide a short "abstain_reason".
- Return output as compact JSON only (no extra text) with this schema:
    {
        "answer": string,                  # concise factual answer (≤150 words)
        "sources": [                       # up to 5 evidence items, ordered by relevance
            {"id": string, "score": number, "excerpt": string}
        ],
        "abstain": boolean,
        "abstain_reason": string|null
    }
- Keep "answer" concise and cite source `id` values in the "sources" array.
"""

QA_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer using ONLY the CONTEXT above.
- Produce output as a single JSON object matching the SYSTEM_PROMPT schema.
- If information is missing, set "abstain": true and provide a short "abstain_reason".
- Limit the answer to ≤150 words and include up to 5 sources with short excerpts (≤200 chars).

Return ONLY the JSON object. No additional commentary.
"""

# ============== Evaluation Metrics ==============
EVAL_METRICS = ["precision", "recall", "f1", "mrr"]
