# Research Paper Question Answering System

A RAG (Retrieval-Augmented Generation) system for answering questions about research papers using efficient small language models.

## 🌟 Features

- **PDF Processing**: Automatically extract text from research papers
- **Semantic Chunking**: Smart text splitting that preserves context
- **Scientific Embeddings**: Uses SPECTER model trained on scientific papers
- **Two-Stage Retrieval**: Dense retrieval + cross-encoder reranking
- **Grounded Generation**: Answers constrained to provided context
- **Source Attribution**: Every answer includes source references
- **Evaluation Metrics**: Built-in precision, recall, and groundedness evaluation

## 🔧 Models Used

| Task | Model | Parameters | Purpose |
|------|-------|------------|---------|
| Embedding | `allenai-specter` | ~110M | Scientific document embeddings |
| Reranking | `ms-marco-MiniLM-L6-v2` | ~22M | Cross-encoder for precise ranking |
| Generation | `Qwen2.5-0.5B-Instruct` | ~490M | Answer generation from context |

**Total: ~622M parameters** (vs 3-7B for typical single-model approaches)

## 📁 Project Structure

```
├── app.py                 # Streamlit web interface
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   ├── uploads/          # PDF upload directory
│   ├── processed/        # Processed documents
│   └── embedding/        # Vector store index
└── src/
    ├── pdf_processor.py  # PDF text extraction
    ├── chunker.py        # Text chunking
    ├── embedder.py       # SPECTER embeddings
    ├── vector_store.py   # FAISS vector storage
    ├── retriever.py      # Retrieval + reranking
    ├── generator.py      # Qwen answer generation
    ├── evaluator.py      # Evaluation metrics
    └── pipeline.py       # Main orchestration
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Reesarch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Programmatic Usage

```python
from src.pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Add documents
pipeline.add_document("path/to/paper.pdf")

# Query
result = pipeline.query("What is the main contribution of this paper?")
print(result.answer)
print(f"Sources: {len(result.sources)}")
print(f"Confidence: {result.confidence:.2%}")
```

## 📊 How It Works

### 1. Document Ingestion
```
PDF → Text Extraction → Metadata Extraction → Section Detection
```

### 2. Indexing
```
Text → Semantic Chunking → SPECTER Embedding → FAISS Index
```

### 3. Query Processing
```
Question → Embed Query → Dense Retrieval → Cross-Encoder Rerank → Top-K Chunks
```

### 4. Answer Generation
```
Context + Question → Qwen2.5-0.5B → Constrained Answer → Source Attribution
```

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Models
EMBEDDING_MODEL = "sentence-transformers/allenai-specter"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Retrieval
TOP_K_RETRIEVAL = 10  # Initial candidates
TOP_K_RERANK = 5      # After reranking

# Chunking
CHUNK_SIZE = 512      # Target chunk size (tokens)
CHUNK_OVERLAP = 50    # Overlap between chunks
```

## 📈 Evaluation

The system includes built-in evaluation metrics:

```python
# Evaluate on test queries
test_queries = [
    {"question": "What is the methodology?", "relevant_doc_ids": ["doc1"]},
    {"question": "What are the results?", "relevant_doc_ids": ["doc1"]}
]

report = pipeline.evaluate(test_queries)
```

### Metrics
- **Precision@K**: Fraction of relevant docs in top-K
- **Recall@K**: Fraction of relevant docs retrieved
- **MRR**: Mean Reciprocal Rank
- **Groundedness**: Answer grounding in context

## 🔬 Design Decisions

### Why Small Models?

1. **Task-Specific Optimization**: Each model is trained for its specific task
2. **Efficiency**: Total ~622M params vs 3-7B for a single large model
3. **Cost-Effective**: Can run on CPU or modest GPU
4. **Lower Latency**: Faster inference for interactive use

### Why SPECTER for Embeddings?

SPECTER is specifically trained on scientific papers using citation graphs, making it ideal for:
- Research paper similarity
- Scientific concept matching
- Cross-paper retrieval

### Why Two-Stage Retrieval?

1. **Stage 1 (Dense)**: Fast approximate retrieval using embeddings
2. **Stage 2 (Rerank)**: Precise scoring with cross-encoder

This provides both speed and accuracy.

### Hallucination Mitigation

- Explicit instructions to only use context
- Low temperature (0.3) for focused generation
- Abstention when information is missing
- Source attribution for transparency

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in embedding
- Use CPU instead of GPU
- Reduce `CHUNK_SIZE`

### Slow Performance
- Enable GPU if available
- Reduce `TOP_K_RETRIEVAL`
- Pre-load models

### Poor Retrieval
- Increase `TOP_K_RETRIEVAL`
- Try hybrid retrieval (dense + BM25)
- Adjust chunk size

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [SPECTER](https://github.com/allenai/specter) from Allen AI
- [Qwen](https://github.com/QwenLM/Qwen2.5) from Alibaba
- [FAISS](https://github.com/facebookresearch/faiss) from Meta