.PHONY: format lint typecheck test eval eval-view clean fetch embed artifacts rebuild \
        staging staging-down staging-logs staging-prune

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

# Static type checking (requires the dev extra: uv sync --extra dev)
typecheck:
	uv run pyright

# Unit tests: no API key, no Redis, no search service — everything is stubbed.
test:
	uv run --group test pytest -q

# The guardrail eval. Unlike `test`, this calls DeepSeek for real — one cheap
# classifier call per case in evals/guard/cases.yaml — so it needs
# DEEPSEEK_API_KEY (in .env, or exported) and costs a fraction of a cent.
# PROMPTFOO_PYTHON is not optional: promptfoo shells out to `python`, which on
# most boxes is either missing or the wrong interpreter. PROMPTFOO_PYTHON_WORKERS=1
# because promptfoo's pool of persistent Python workers (one per concurrent
# case by default) stalls on this box — cases time out after 5 minutes while
# the classifier itself answers in a second. One worker runs the set in ~40s.
eval:
	cd evals/guard && PROMPTFOO_PYTHON=$(CURDIR)/.venv/bin/python PROMPTFOO_PYTHON_WORKERS=1 npx -y promptfoo@latest eval --no-cache

# The last run's results as a browsable table.
eval-view:
	cd evals/guard && npx -y promptfoo@latest view --yes

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

# ---- Staging on this box: https://hn.staging.fiodorov.es (GitHub login) ----
# Builds the image from the working tree — no commit, no push, no Railway. The
# shared Caddy + oauth2-proxy edge lives in ../staging-infra and must be up
# first; this brings up only the app and its own Redis, neither publishing a port.
STAGING := docker compose -f docker-compose.staging.yml

staging:
	@docker network inspect staging >/dev/null 2>&1 || \
		{ echo "no 'staging' network — run 'make up' in ../staging-infra first"; exit 1; }
	$(STAGING) up -d --build

staging-down:
	$(STAGING) down

staging-logs:
	$(STAGING) logs -f hn-search

# Every `make staging` orphans the previous image and grows the build cache.
# Safe: nothing in use is removed.
staging-prune:
	docker image prune -f
	docker builder prune -f
