#!/bin/bash
# Build the binary-quantized HNSW index on all hn_documents* tables.
#
# This replaces the old float (vector_cosine_ops) HNSW index, which needed the
# full ~37 GB of vectors resident in RAM. The binary index is ~10x smaller
# (~3.7 GB at 9.4M rows) and search reranks a Hamming shortlist by exact cosine
# (see hn_search/rag/nodes.py). Connects via DATABASE_URL / PG* env vars, so it
# works locally or against Railway.
#
# Usage:
#   ./misc/optimize_and_create_index.sh              # build/upgrade indexes
#   ./misc/optimize_and_create_index.sh --concurrent # no write-lock (slower)
#   MAINTENANCE_WORK_MEM=2GB ./misc/optimize_and_create_index.sh

set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python misc/build_binary_index.py "$@"
