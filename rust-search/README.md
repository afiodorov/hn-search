# rust-search

Standalone brute-force vector search for hn-search. Replaces the Postgres/pgvector
path with: **binary-quantized Hamming shortlist → float16 exact-cosine rerank →
SQLite text lookup**, over an mmap'd immutable base plus an appendable tail for daily
online updates. Working set at ~18M rows is ~1.7 GB (the hot binary codes).

See the design doc: `../.claude/plans/i-get-it-now-dreamy-hedgehog.md`.

## Artifacts

Built by `misc/build_search_artifacts.py` (row-aligned, ordered by `id::bigint`):

| file             | shape            | role                                  |
|------------------|------------------|---------------------------------------|
| `codes.bin`      | N × 96 B         | sign-bit codes — hot, mmap, scanned   |
| `rerank_f16.bin` | N × 768 × 2 B    | f16 vectors — mmap, only shortlist read|
| `docs.sqlite`    | `doc(rowid,...)` | text/metadata; rowid = row index + 1  |
| `meta.json`      | `{count, dim}`   | integrity / base count                |

Appended rows live in `tail_codes.bin` / `tail_f16.bin` (+ rows in `docs.sqlite`);
the service folds them into the base on rebuild.

## HTTP API

`Authorization: Bearer $HN_SEARCH_TOKEN` required for all but `/health`.

- `GET  /health` → `ok`
- `POST /search` `{embedding:[f32;768], k?, shortlist?}` → `[{id, clean_text, author, timestamp, type, distance}]` (asc distance)
- `POST /append` `{rows:[{hn_id, clean_text, author, timestamp, type, embedding}]}` → `{appended, skipped, max_id}` (dedup by hn_id)
- `GET  /max_id` → `{max_id}` (resume point for the daily updater)

## Build & run locally

```sh
cargo build --release
ARTIFACT_DIR=./artifacts PORT=8001 HN_SEARCH_TOKEN=dev ./target/release/rust-search
```

Env: `ARTIFACT_DIR` (default `artifacts`), `PORT` (8001), `HN_SHORTLIST` (200),
`HN_SEARCH_TOKEN` (auth disabled if unset).

## Bootstrap → deploy

```sh
# 1. one-time bootstrap dump from the live Postgres
uv run python misc/build_search_artifacts.py --out artifacts/

# 2. QA parity (Gate B) against a locally-running service
uv run python misc/eval_rust_parity.py --artifacts artifacts/ --url http://localhost:8001

# 3. ship to the Hetzner VPS (atomic symlink flip + restart)
REMOTE=hnsearch@your-vps ./scripts/rsync_artifacts.sh artifacts/
```

Deploy files in `deploy/`: `hnsearch.service` (systemd, reads `/etc/hnsearch.env`),
`Caddyfile` (auto-HTTPS reverse proxy). Build the binary on the VPS (or scp a
`--release` build to `/usr/local/bin/rust-search`).

## Daily updates

`misc/cron_update.py` with `HN_UPDATE_TARGET=rust` (and `HN_SEARCH_URL` /
`HN_SEARCH_TOKEN`) fetches new comments from BigQuery, embeds them with the CPU ONNX
encoder (no GPU), and POSTs to `/append`. New rows are searchable immediately via the
tail; no index rebuild.

## Cut-over (web app)

Set on the Railway app: `HN_SEARCH_BACKEND=shadow` (compare pg vs rust on live
traffic, log overlap), then `rust` once stable. `HN_SEARCH_URL`, `HN_SEARCH_TOKEN`
point at this service. See `hn_search/search_backend.py`.
