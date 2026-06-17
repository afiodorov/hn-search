# CLAUDE.md

Semantic search + RAG over ~12.1M Hacker News comments. A FastAPI/LangGraph web app
(Railway) talks to a hand-rolled Rust vector-search service (Hetzner VPS) over HTTPS.
Read `README.md` and `rust-search/README.md` first — they hold the architecture,
recall numbers, and cost rationale. This file is the operator's cheat-sheet.

## Layout

- `hn_search/` — Python web app: FastAPI + SSE API, LangGraph RAG, ONNX query encoder,
  Redis cache, `search_backend.py` (talks to the Rust service).
- `rust-search/` — the search service (axum, rayon, memmap2, rusqlite, half).
  `src/main.rs` is the HTTP layer; `index.rs`/`quantize.rs`/`db.rs` do the work.
- `misc/` — data pipeline: BigQuery fetch, GPU/CPU embedding, artifact build, the
  daily incremental updater.
- `frontend/` — React UI; `npm run build` emits assets the API serves.

## Two embedding paths (don't confuse them)

- **Serve** (default deps, `uv sync`): query encoding via **ONNX Runtime, no torch**
  (~300 MB RAM). This is what the web app uses.
- **Batch** (`uv sync --extra dev` / `uv run --extra dev`): torch +
  sentence-transformers for embedding the corpus or new comments. Model is
  `all-mpnet-base-v2`, **768-d**.

## The Rust service

Two-stage retrieval over mmap'd flat files + an appendable in-memory tail:
binary-quantized Hamming shortlist (`codes.bin`, hot) → f16 exact-cosine rerank
(`rerank_f16.bin`, cold mmap) → text from `docs.sqlite`. Daily `/append` rows land in
the tail and are searchable immediately, no rebuild.

Endpoints (all but `/health` need `Authorization: Bearer`):
`GET /health`, `POST /search`, `POST /append`, `GET /max_id`.

Auth is **two-token**: `HN_SEARCH_TOKEN` (read → `/search`) and
`HN_SEARCH_ADMIN_TOKEN` (write → `/append`, `/max_id`). Admin is a superset.

### Request body limits

`/append` ships 768-d f32 embeddings as JSON (~10 KB/row). Both Caddy *and* axum cap
bodies — they must agree:
- Caddy: `request_body max_size 64MB` in the Caddyfile.
- axum: defaults to **2 MB**; `main.rs` lifts the `/append` route to 64 MB via
  `DefaultBodyLimit::max(...)`. (A 413 with Caddy already at 64 MB means axum's cap.)

The daily updater batches at 1000 rows (~10 MB), comfortably under 64 MB.

## Production deployment (Hetzner VPS)

- **Host**: `root@167.233.115.172`, public URL `https://167.233.115.172.sslip.io`
  (sslip.io maps the IP to a hostname so Caddy can issue a Let's Encrypt cert).
- **Service**: systemd unit `hnsearch` → `/usr/local/bin/rust-search`, reading
  `ARTIFACT_DIR=/var/lib/hnsearch/current` (a symlink) and `/etc/hnsearch.env`
  (holds the tokens). Unit source: `rust-search/deploy/hnsearch.service`.
- **TLS/proxy**: Caddy, `/etc/caddy/Caddyfile` → `reverse_proxy 127.0.0.1:8001`.
- **Build**: the box has cargo and a plain copy of the `rust-search/` subdir at
  `/root/rust-search` (no git). The binary is built **on the box**.

### Deploy a code change to the Rust service

```sh
# from repo root — sync source (NOT target), build on the box, install, restart
rsync -avP --exclude target --exclude .git \
  rust-search/src rust-search/Cargo.toml rust-search/Cargo.lock \
  root@167.233.115.172:/root/rust-search/
ssh root@167.233.115.172 '
  cd /root/rust-search && cargo build --release &&
  install -m755 target/release/rust-search /usr/local/bin/rust-search &&
  systemctl restart hnsearch && sleep 2 &&
  journalctl -u hnsearch --no-pager -n 3'
```

The startup log line `loaded base=… tail=… max_id=… read_auth=… admin_auth=…`
confirms it came up. The **tail persists across restarts** (`tail_codes.bin` /
`tail_f16.bin` + rows in `docs.sqlite`), so a redeploy never loses appended data.

### Ship new artifacts (full rebuild only)

Use `rust-search/scripts/rsync_artifacts.sh` (atomic release dir + symlink flip).
A fresh base ships with an empty tail; the next updater re-appends `id > max_id`
from BigQuery. Only flip a release built from a complete, up-to-date dump.

## Updating the corpus

**Daily incremental** (run where the admin token lives — laptop):
```sh
uv run --extra dev python misc/fetch_and_embed_new_comments.py
```
Reads the service's `/max_id` → pulls newer comments from BigQuery → embeds on CPU
(ONNX) → POSTs to `/append`. Flags: `--skip-fetch`, `--skip-embed`, `--skip-append`,
`--reset`. It checkpoints to `data/raw/fetch_state.json` and reuses existing
raw/embedded parquet files, so a re-run after a failure resumes cheaply.

**Full rebuild** (rare): `make fetch` → `make embed` (on a GPU box) → `make artifacts`
→ `rsync_artifacts.sh`. See README.

## Dev

```sh
uv sync                                  # serve deps
uv run python -m hn_search.api           # API + UI on :8000
cd rust-search && cargo build --release  # local service
ARTIFACT_DIR=./artifacts PORT=8001 HN_SEARCH_TOKEN=dev ./target/release/rust-search
```

- Format/lint: `make format`, `make lint` (ruff). Line length 88, target py313.
- Auth is **disabled** when no token env is set (local dev only).
