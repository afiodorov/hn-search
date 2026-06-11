# 🔎 HN Search: Semantic Search & RAG for Hacker News

A production-ready semantic search engine and RAG (Retrieval-Augmented Generation) system for Hacker News comments, built with vector embeddings, PostgreSQL with pgvector, and LangGraph.

## 🎯 Project Overview

This project implements a full-stack semantic search system over **millions of Hacker News comments** (2023-2025), enabling natural language queries and AI-powered question answering. Unlike traditional keyword search, it uses dense vector embeddings to understand semantic meaning, finding relevant discussions even when exact keywords don't match.

**Live Demo**:

Hosted at [hn.fiodorov.es](https://hn.fiodorov.es)

**Key Features:**
- 🔍 **Semantic Search**: Natural language queries with cosine similarity ranking
- 🤖 **RAG System**: AI-powered Q&A using LangGraph workflows and DeepSeek LLM
- 🪶 **RAM-light vector search**: binary-quantized HNSW + exact rerank — ~10× smaller hot index than float, no measurable quality loss
- ⚡ **ONNX query encoder**: serve without torch (~300 MB instead of ~1.5 GB RAM)
- ⚡ **Redis Caching**: Sub-second response times for repeated queries
- 🔄 **Incremental Updates**: Idempotent data pipeline for fetching new HN comments
- 🎨 **Web Interface**: Gradio-based UI with URL parameter support
- 🏗️ **Partitioned Tables**: Time-based partitioning for efficient query performance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  BigQuery (HN Public Dataset)                                   │
│         ↓                                                        │
│  Fetch New Comments (idempotent, resumable)                     │
│         ↓                                                        │
│  Generate Embeddings (sentence-transformers, MPS/CUDA)          │
│         ↓                                                        │
│  PostgreSQL + pgvector (partitioned by month)                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Query System                             │
├─────────────────────────────────────────────────────────────────┤
│  User Query                                                      │
│         ↓                                                        │
│  Encode with all-mpnet-base-v2 via ONNX Runtime (CPU, no torch) │
│         ↓                                                        │
│  Redis Cache Check ────────────────┐                            │
│         ↓                           │ (cache hit)                │
│  Binary-quant Hamming shortlist → exact cosine rerank          │
│         ↓                           │                            │
│  Cache Results ─────────────────────┘                            │
│         ↓                                                        │
│  Return Top K Documents                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         RAG System                               │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph Workflow (StateGraph)                                │
│         ↓                                                        │
│  [Retrieve] → Vector Search → Top 10 Comments                    │
│         ↓                                                        │
│  [Answer] → DeepSeek LLM → Generated Response                   │
│         ↓                                                        │
│  Gradio Web UI (with sources & citations)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technical Stack

### Core Technologies
- **Language**: Python 3.13
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (fast Python package installer)
- **Database**: PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector) extension (`halfvec` + binary quantization)
- **Vector Model**: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (768-dim) — torch for batch embedding, **ONNX Runtime** at serve time
- **LLM**: DeepSeek via OpenAI-compatible API
- **Cache**: Redis 6.x
- **Data Source**: [BigQuery HN Public Dataset](https://console.cloud.google.com/marketplace/product/y-combinator/hacker-news)

### Key Libraries
- **sentence-transformers**: High-quality sentence embeddings
- **psycopg3**: Modern PostgreSQL adapter with async support
- **pgvector**: PostgreSQL extension for vector similarity search
- **LangGraph**: Orchestration framework for LLM workflows
- **LangChain**: LLM abstractions and prompting
- **Gradio**: Web UI framework
- **torch**: PyTorch for model inference (MPS support for Apple Silicon)
- **pandas + pyarrow**: Data processing pipeline

### Infrastructure
- **Compute**: Railway (PostgreSQL + Redis) / Local development
- **Deployment**: Docker-ready with docker-compose.yml

## 📊 Dataset

**Source**: Hacker News comments from BigQuery public dataset (`bigquery-public-data.hacker_news.full`)

**Scope**:
- Time range: January 2023 - September 2025
- Comment count: ~9.4M comments
- Filters: Non-deleted, non-dead, non-null text

**Partitioning Strategy**:
- Tables partitioned by month (e.g., `hn_documents_2023_01`, `hn_documents_2023_02`, ...)
- Enables efficient querying and index management
- Simplifies incremental updates

## 🚀 Getting Started

### Prerequisites

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv

# Install PostgreSQL with pgvector
# See: https://github.com/pgvector/pgvector#installation

# Install Redis (optional, for caching)
brew install redis  # macOS
sudo apt install redis  # Ubuntu
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hn-search.git
cd hn-search

# Install dependencies
uv sync

# For development (includes BigQuery tools)
uv sync --extra dev

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials:
#   DATABASE_URL=postgres://user:pass@host:port/dbname
#   DEEPSEEK_API_KEY=sk-...
#   REDIS_URL=redis://localhost:6379  (optional)
```

### Database Setup

```bash
# Initialize database with precomputed embeddings (creates the binary index)
uv run python -m hn_search.init_db_pgvector

# Or initialize with test mode (100 docs only)
uv run python -m hn_search.init_db_pgvector --test

# (Re)build the binary HNSW index on all hn_documents* tables — also migrates any
# vector(768) columns to halfvec(768). Idempotent; connects via DATABASE_URL.
uv run python misc/build_binary_index.py
# add --concurrent to avoid write locks on a live database
```

### Full Rebuild From Scratch

Bringing the corpus up from nothing (fresh BigQuery pull → GPU embeddings →
partitions). Each step is **resumable** and **month-aligned**: one
`month_YYYY_MM.parquet` per month maps to one `hn_documents_YYYY_MM` partition.

**Prerequisites**: a Postgres+pgvector database with `DATABASE_URL` set in `.env`,
and GCP credentials for BigQuery (`GOOGLE_CLOUD_PROJECT` / `GOOGLE_APPLICATION_CREDENTIALS_JSON`).

```bash
# 1. Fetch comments from BigQuery, one parquet per month (cheap, runs anywhere).
#    Defaults to 2023-01 .. current month; override with --start/--end YYYY-MM.
make fetch                 # → data/raw/month_*.parquet

# 2. Embed on a rented GPU box (e.g. vast.ai). `make embed` runs LOCALLY on whatever
#    machine you invoke it on, using that machine's GPU — so it runs ON the box, not
#    from your laptop. The helper below drives it from your laptop end to end: it
#    pushes code + data/raw (NO secrets — embedding needs none), installs CUDA torch,
#    and starts the embed in a remote tmux session (survives disconnects, resumable).
REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22 ./misc/gpu_embed.sh
#    Then pull finished months back as they complete:
REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22 \
  REMOTE_DIR=/root/hn-search/data/embedded ./misc/sync_embeddings.sh
#    (Or, if you're already on the box: just run `make embed`.)

# 3. Load embedded months into partitions (halfvec + binary HNSW index per month).
#    Run from wherever has DATABASE_URL (loading box -> Railway directly is fastest).
make load                  # populates hn_documents_YYYY_MM, builds binary indexes

# 4. Attach the monthly tables under one partitioned parent `hn_documents`, so the
#    web path queries them in a single MergeAppend. Idempotent — re-run after new
#    months are added by the incremental fetcher.
make attach
```

The GPU is only needed for step 2, and that box needs **no credentials** — only the
public model (auto-downloaded from HuggingFace) and your raw parquet files. Destroy the
instance once `data/embedded` is synced back. `misc/fetch_and_embed_new_comments.py`
then keeps the corpus current with incremental monthly pulls; serving uses ONNX (no torch).

## 💡 Usage

### 1. Semantic Search (CLI)

```bash
# Basic search
uv run python -m hn_search.query "What do people think about Rust vs Go?"

# Get more results
uv run python -m hn_search.query "best practices for system design" 20

# Output includes:
# - Comment ID (with HN link)
# - Author
# - Timestamp
# - Cosine distance score
# - Full comment text
```

### 2. RAG System (Web UI)

```bash
# Start the Gradio web interface
uv run python -m hn_search.rag.web_ui

# Open http://localhost:7860
# Ask questions like:
#   "What are the main criticisms of microservices?"
#   "How do people debug production issues?"
#   "What do HN users think about AI coding assistants?"
```

**Features**:
- Real-time streaming responses
- Source citations with HN links
- URL parameter support: `?q=your+question`
- Auto-search from URL parameters

### 3. Incremental Data Updates

```bash
# Fetch new comments, generate embeddings, and upsert to DB
uv run --extra dev python misc/fetch_and_embed_new_comments.py

# Options:
#   --project <GCP_PROJECT>  # Specify BigQuery billing project
#   --skip-fetch             # Skip BigQuery download
#   --skip-embed             # Skip embedding generation
#   --skip-upsert            # Skip database insertion
#   --reset                  # Clear state and start fresh

# Resume interrupted runs (automatic)
uv run --extra dev python misc/fetch_and_embed_new_comments.py

# The script is fully idempotent and resumable:
# - Saves state to data/raw/fetch_state.json
# - Checks for existing files before re-downloading
# - Incremental embedding generation with checkpoints
# - Tracks processed IDs to avoid duplicate inserts
```

### 4. Generate Embeddings (Batch)

A single 5090 Nvidia was rented from vast.ai to compute all historical
embeddings for a few dollars.

```bash
# Process raw parquet files and generate embeddings (torch lives in the dev extra)
uv run --extra dev python misc/generate_embeddings_gpu.py

# Uses MPS (Apple Silicon) or CUDA automatically
# Processes in batches to avoid OOM
# Saves to embeddings/*.parquet
```

## 🔍 How It Works

### Vector Search

The system uses **binary-quantized HNSW with exact reranking** — a two-stage search
that keeps the hot index ~10× smaller than a plain float index:

```sql
-- Stage 1: shortlist by Hamming distance on the binary index (RAM-resident).
-- Stage 2: rerank the shortlist by exact cosine distance.
SELECT id, clean_text, author, timestamp, type, distance
FROM (
    SELECT id, clean_text, author, timestamp, type,
           embedding <=> :query AS distance
    FROM hn_documents
    ORDER BY binary_quantize(embedding)::bit(768)
             <~> binary_quantize(:query)::bit(768)
    LIMIT 200
) shortlist
ORDER BY distance
LIMIT 10;
```

`binary_quantize` keeps one bit per dimension (the sign), so the searchable index
stores 96 bytes/vector instead of 3072. Hamming search narrows millions of rows to
a 200-row shortlist using the tiny index; exact cosine then reranks just those. On
this corpus that recovers full result quality — strict recall@10 ≈ 0.90, and
**tie-tolerant recall@10 = 0.999** (the "misses" are alternative neighbors at
identical cosine distance, e.g. duplicate comments).

**Performance Optimizations**:
- Binary-quantized HNSW index (`bit_hamming_ops`) + exact rerank — ~3.7 GB hot
  index at 9.4M rows vs ~37 GB for a float HNSW index
- `halfvec(768)` storage — half the on-disk vector size, no quality loss
- ONNX Runtime query encoder — ~300 MB RAM vs ~1.5 GB for torch, no torch at serve
- Redis caching layer reduces repeated queries to <100ms
- Connection pooling with psycopg3
- Native monthly range partitioning: a single query against the parent
  `hn_documents` fans out across each partition's binary index via a Postgres
  `MergeAppend` (verified in the plan) — no application-side fan-out

### RAG Pipeline

The RAG system uses LangGraph to orchestrate a two-node workflow:

1. **Retrieve Node**:
   - Encodes user query with sentence-transformers
   - Performs vector search in PostgreSQL
   - Returns top 10 most relevant comments

2. **Answer Node**:
   - Formats retrieved comments as context
   - Prompts DeepSeek LLM with query + context
   - Streams response back to user

**Prompt Engineering**:
```python
system_prompt = """You are a helpful assistant that answers questions
based on Hacker News discussions. Use the provided comments to give
accurate, well-sourced answers. Cite comment numbers [1], [2], etc."""

user_prompt = f"""Question: {query}

Context from HN comments:
{formatted_comments}

Answer:"""
```

### Embedding Model

**Model**: `sentence-transformers/all-mpnet-base-v2`
- Dimensions: 768
- Max sequence length: 384 tokens
- Training: MS MARCO + Natural Questions + other datasets
- Performance: SOTA for semantic similarity tasks

**Why this model?**
- Excellent balance of quality vs. speed
- Pre-trained on diverse Q&A datasets
- Good generalization to HN comment domain
- Efficient inference on CPU/MPS/CUDA

**Serving the model cheaply**: the corpus is embedded offline (torch, on a rented
GPU), but at serve time only the user's *query* is embedded. That runs through
[ONNX Runtime](https://onnxruntime.ai/) using the pre-exported ONNX weights shipped
in the model's HF repo — verified bit-exact (cosine 1.0) against both
sentence-transformers and the stored corpus embeddings, so no torch is needed in the
serving image. On ARM hosts set `HN_ONNX_MODEL_FILE=onnx/model_qint8_arm64.onnx`.

## 📈 Performance & Scale

### Current Scale
- **Documents**: ~9.4M Hacker News comments
- **Storage**: ~25 GB (`halfvec` embeddings + text + binary index)
- **Hot index**: ~3.7 GB binary HNSW (vs ~37 GB float) — the RAM-cost win
- **Vector search**: ~2-3 ms/query (Hamming shortlist + exact rerank)
- **Query Latency**:
  - Cold query: ~30s (embedding + search + LLM)
  - Cached query: <1s (Redis cache hit)
  - Concurrent duplicate query: <1s (job deduplication)
  - RAG end-to-end: ~30s (including LLM generation)

### Production Optimizations ⚡

**Implemented**:
1. **Singleton Embedding Model**: Model loaded once and reused (3-5x throughput)
2. **Job Deduplication**: Concurrent duplicate queries share processing (saves 90%+ compute)
3. **Multi-layer Caching**: Redis cache for vector search, LLM answers, and job results
4. **Connection Pooling**: PostgreSQL connection pool (min: 2, max: 20)
5. **Partitioned Tables**: Monthly partitions for efficient indexing
6. **Incremental Updates**: Only process new comments since last run

### Capacity

**Single Instance**:
- 20-30 concurrent users (unique queries)
- 100+ concurrent users (with 80% cache hit rate)

**Horizontal Scaling** (Railway/Cloud):
- 2 replicas: 40-60 concurrent users
- 4 replicas: 80-120 concurrent users
- 8 replicas: 160-240 concurrent users

See [RAILWAY.md](RAILWAY.md) for deployment guide.

### Resource Requirements
- **App RAM**: ~0.3-0.5 GB per instance (ONNX query encoder; no torch)
- **PostgreSQL RAM**: hot working set ≈ the binary index (~3.7 GB at 9.4M),
  vs ~37 GB for a float HNSW index — fits a small instance. This is the main
  cost win: it cuts the Railway bill from ~$50-100/mo to a small-instance tier.
- **Storage**: ~25 GB for database (`halfvec` vectors + text + binary index)
- **CPU**: 0.5-1.0 cores per instance
- **PostgreSQL**: 100+ connections (20 per instance)
- **Redis**: 512MB-1GB for cache

## 🔧 Configuration

### Environment Variables

```bash
# Required
DATABASE_URL=postgres://user:pass@host:port/dbname
DEEPSEEK_API_KEY=sk-...

# Optional
REDIS_URL=redis://localhost:6379
GOOGLE_CLOUD_PROJECT=your-gcp-project
TOKENIZERS_PARALLELISM=false  # Disable for multi-threaded use
```

### PostgreSQL Settings

For optimal performance on Railway/cloud instances:
```sql
-- Default settings (already applied)
shared_buffers = 128MB
maintenance_work_mem = 64MB
work_mem = 4MB
max_parallel_workers = 8
effective_cache_size = 4GB
```

## 🧪 Development

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Sort imports
make imports
```

### Project Structure

```
hn-search/
├── hn_search/              # Main package
│   ├── query.py           # Vector search interface
│   ├── init_db_pgvector.py # Database initialization
│   ├── db_config.py       # Database connection config
│   ├── cache_config.py    # Redis caching layer
│   ├── common.py          # Shared utilities
│   └── rag/               # RAG system
│       ├── graph.py       # LangGraph workflow
│       ├── nodes.py       # Retrieve & Answer nodes
│       ├── state.py       # State management
│       ├── cli.py         # CLI interface
│       └── web_ui.py      # Gradio web interface
├── misc/                   # Utility scripts
│   ├── generate_embeddings_gpu.py  # Batch embedding generation
│   └── fetch_and_embed_new_comments.py  # Incremental updates
├── data/                   # Data directory
│   └── raw/               # Raw parquet files
├── pyproject.toml         # Project dependencies
└── Makefile               # Development shortcuts
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Vector Search at Scale**: Implementing semantic search with pgvector on millions of documents
2. **Production ML Pipelines**: Idempotent, resumable data processing with checkpointing
3. **RAG Architecture**: Building retrieval-augmented generation with LangGraph
4. **Database Optimization**: Partitioning strategies, connection pooling, caching
5. **Modern Python Tooling**: uv, ruff, type hints, async patterns
6. **Cloud Integration**: BigQuery public datasets, Railway deployment, Redis caching
7. **GPU Optimization**: MPS/CUDA support for efficient embedding generation

## 📚 References

- [pgvector: Open-source vector similarity search for Postgres](https://github.com/pgvector/pgvector)
- [sentence-transformers: State-of-the-art sentence embeddings](https://www.sbert.net/)
- [LangGraph: Building stateful, multi-actor LLM applications](https://langchain-ai.github.io/langgraph/)
- [BigQuery HN Dataset](https://console.cloud.google.com/marketplace/product/y-combinator/hacker-news)
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/hn-search.git

# Create a branch
git checkout -b feature/your-feature

# Make changes and test
uv run python -m hn_search.query "test query"

# Format and lint
make format

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature
```

## 🙏 Acknowledgments

- Y Combinator for open-sourcing Hacker News data
- The pgvector team for excellent PostgreSQL integration
- sentence-transformers community for pre-trained models
- LangChain team for RAG tooling

---

**⭐ If you find this project useful, please consider giving it a star!**

![example](./example.png)
