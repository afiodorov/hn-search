# 🔎 HN Search: Semantic Search & RAG for Hacker News

A production-ready semantic search engine and RAG (Retrieval-Augmented Generation) system for Hacker News comments, built with vector embeddings, PostgreSQL with pgvector, and LangGraph.

## 🎯 Project Overview

This project implements a full-stack semantic search system over **millions of Hacker News comments** (2023-2025), enabling natural language queries and AI-powered question answering. Unlike traditional keyword search, it uses dense vector embeddings to understand semantic meaning, finding relevant discussions even when exact keywords don't match.

**Live Demo**:

Hosted at [hn.fiodorov.es](https://hn.fiodorov.es)

**Key Features:**
- 🔍 **Semantic Search**: Natural language queries with cosine similarity ranking
- 🤖 **RAG System**: AI-powered Q&A using LangGraph workflows and DeepSeek LLM
- 📊 **Scalable Architecture**: PostgreSQL with pgvector for production-grade vector search
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
│  Encode with sentence-transformers/all-mpnet-base-v2            │
│         ↓                                                        │
│  Redis Cache Check ────────────────┐                            │
│         ↓                           │ (cache hit)                │
│  PostgreSQL Vector Search (cosine) │                            │
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
- **Database**: PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector) extension
- **Vector Model**: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (768-dim embeddings)
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
# Initialize database with precomputed embeddings
uv run python -m hn_search.init_db_pgvector

# Or initialize with test mode (100 docs only)
uv run python -m hn_search.init_db_pgvector --test
```

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
# Process raw parquet files and generate embeddings
uv run python misc/generate_embeddings_gpu.py

# Uses MPS (Apple Silicon) or CUDA automatically
# Processes in batches to avoid OOM
# Saves to embeddings/*.parquet
```

## 🔍 How It Works

### Vector Search

The system uses cosine distance for similarity search:

```sql
SELECT id, clean_text, author, timestamp, type,
       embedding <=> query_vector AS distance
FROM hn_documents
ORDER BY embedding <=> query_vector
LIMIT 10
```

**Performance Optimizations**:
- HNSW index for approximate nearest neighbor search
- Redis caching layer reduces repeated queries to <100ms
- Connection pooling with psycopg3
- Partitioned tables for efficient index scans

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

## 📈 Performance & Scale

### Current Scale
- **Documents**: ~9.4M Hacker News comments
- **Storage**: ~40 GB (including embeddings)
- **Query Latency**:
  - Cold query: ~30s (embedding + search)
  - Cached query: ~50ms
  - RAG end-to-end: ~60s (including LLM generation)

### Optimization Techniques
1. **Caching**: Redis cache for vector search results (90%+ hit rate for common queries)
2. **Batching**: Batch embedding generation (128 docs/batch) for GPU efficiency
3. **Partitioning**: Monthly table partitions for faster queries and easier maintenance
4. **Connection Pooling**: psycopg3 connection pool reduces overhead
5. **Incremental Updates**: Only process new comments since last run

### Resource Requirements
- **RAM**: 2-4 GB for query service, 8-16 GB for embedding generation
- **Storage**: 1 GB per ~70k comments (including embeddings)
- **GPU**: Optional (MPS/CUDA) for 5-10x faster embedding generation

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

## 🚧 Future Improvements

### Short Term
- [ ] Add HNSW index for faster approximate nearest neighbor search
- [ ] Implement query rewriting for better retrieval
- [ ] Add user feedback loop to improve results
- [ ] Deploy to production with auto-scaling

### Medium Term
- [ ] Multi-query retrieval with reciprocal rank fusion
- [ ] Fine-tune embeddings on HN-specific data
- [ ] Add hybrid search (vector + BM25)
- [ ] Implement query expansion with LLM

### Long Term
- [ ] Real-time indexing of new HN comments via Algolia/RSS
- [ ] Multi-modal search (include linked articles)
- [ ] Personalized search based on user preferences
- [ ] Community-driven relevance ratings

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
