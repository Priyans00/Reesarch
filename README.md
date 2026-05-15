# Reesarch — Research Paper Question Answering

A lightweight, reproducible Retrieval-Augmented Generation (RAG) pipeline for answering questions about research papers.

**Highlights**
- Designed for accuracy and efficiency: small, task-specific models and two-stage retrieval.
- Grounded answers with source attribution and evaluation utilities.

**Quick Start**

- Clone and install:

```bash
git clone https://github.com/Priyans00/Reesarch
cd reesarch
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

- Run the web UI:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Programmatic use**

```python
from src.pipeline import create_pipeline
pipeline = create_pipeline()
pipeline.add_document("path/to/paper.pdf")
result = pipeline.query("What is the main contribution?")
print(result.answer)
print("Sources:", len(result.sources))
```

**Repository layout**

```
app.py
config.py
requirements.txt
data/         # uploads, processed artifacts, vector index
src/          # core modules: pdf_processor, chunker, embedder, vector_store, retriever, generator, pipeline
```

**Configuration**
- See `config.py` for the main runtime options (models, retrieval sizes, chunking).

Recommended sensible defaults are included; modify values only if you understand the downstream effects.

**Design notes**
- Embeddings: SPECTER (scientific-text tuned) for document similarity.
- Retrieval: dense vector search followed by a cross-encoder reranker for precision.
- Generation: constrained, low-temperature decoding with explicit grounding to retrieved context.

**Troubleshooting**
- Out of memory: reduce embedding batch sizes or CHUNK_SIZE; fall back to CPU.
- Slow: enable GPU, lower TOP_K_RETRIEVAL, or pre-load models.

**License**
- MIT

**Acknowledgments**
- SentenceTransformers, SPECTER, FAISS, and the Qwen model authors.