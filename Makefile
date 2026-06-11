.PHONY: format lint clean fetch embed load index attach rebuild

# Default target - run both formatting and linting
format:
	uv run ruff check --select I --fix .
	uv run ruff format .

# Just run import sorting
imports:
	uv run ruff check --select I --fix .

# Just run code formatting
fmt:
	uv run ruff format .

# Run linting (without fixes)
lint:
	uv run ruff check .

# Clean ChromaDB container and volumes
clean:
	docker compose down -v

# ---- Full rebuild pipeline (see README "Full Rebuild From Scratch") ----
# 1. Fetch HN comments from BigQuery, one parquet per month (runs anywhere).
fetch:
	uv run --extra dev python misc/fetch_historical.py

# 2. Embed the monthly shards. Runs on THIS machine's GPU — so run it ON the GPU box.
#    To drive a remote box from your laptop instead, use: ./misc/gpu_embed.sh
embed:
	uv run --extra dev python misc/generate_embeddings_gpu.py

# 3. Load embedded months into hn_documents_YYYY_MM partitions (+ binary index).
load:
	uv run python misc/load_embedded_to_partitions.py

# (Re)build / upgrade the binary HNSW index on all partitions.
index:
	uv run python misc/build_binary_index.py

# Attach monthly tables under the partitioned parent `hn_documents` so the web path
# queries them in one MergeAppend. Idempotent; re-run after new months are added.
attach:
	uv run python misc/attach_partitions.py

# Embed + load + attach (run after `make fetch`); embed needs a GPU.
rebuild: embed load attach
