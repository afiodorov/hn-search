.PHONY: format lint clean fetch embed artifacts rebuild

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
# Daily growth goes through the service's /append; this is only for a from-scratch
# rebuild of the artifact files.
# 1. Fetch HN comments from BigQuery, one parquet per month (runs anywhere).
fetch:
	uv run --extra dev python misc/fetch_historical.py

# 2. Embed the monthly shards → data/embedded/*.parquet. Runs on THIS machine's GPU,
#    so run it ON the GPU box. To drive a remote box instead, use: ./misc/gpu_embed.sh
embed:
	uv run --extra dev python misc/generate_embeddings_gpu.py

# 3. Build the flat artifact files from the embedded parquet, then rsync them to the
#    box (rust-search/scripts/rsync_artifacts.sh).
artifacts:
	uv run python misc/build_search_artifacts.py --out rust-search/artifacts

# Embed + build artifacts (run after `make fetch`); embed needs a GPU.
rebuild: embed artifacts
