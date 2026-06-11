#!/bin/bash
# Provision a rented GPU box (e.g. vast.ai) and run the embedding step on it.
#
# Run this on your LOCAL machine. It:
#   1. pushes code + data/raw/month_*.parquet to the box  (NO .env / secrets / DB creds)
#   2. installs uv + deps (CUDA torch) on the box
#   3. starts `generate_embeddings_gpu.py` inside a remote tmux session, so it keeps
#      running if your SSH connection drops. The embed step is resumable (skips months
#      already done), so you can re-run this safely.
#
# Embedding needs NO secrets on the box — just the public model (pulled from HuggingFace)
# and your raw parquet files. The DB load (`make load`) runs back on your laptop.
#
# Assumes a Debian/Ubuntu root image (typical on vast.ai).
#
# Usage:
#   REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22 ./misc/gpu_embed.sh
#
# Watch progress:   ssh -p PORT root@IP 'tmux attach -t embed'      (detach: Ctrl-b d)
# Pull results:     REMOTE_HOST=root@IP REMOTE_PORT=PORT \
#                     REMOTE_DIR=/root/hn-search/data/embedded ./misc/sync_embeddings.sh
# Finish (laptop):  make load

set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST=user@ip (from vast.ai)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/root/hn-search}"
RSH="ssh -p $REMOTE_PORT"
SSH="ssh -p $REMOTE_PORT $REMOTE_HOST"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> 1/3 pushing code + raw shards to $REMOTE_HOST:$REMOTE_DIR (no secrets)"
rsync -az -e "$RSH" \
  --exclude '.git' --exclude '.venv' --exclude '.env' \
  --exclude 'data' --exclude '__pycache__' --exclude '.ruff_cache' \
  "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
$SSH "mkdir -p '$REMOTE_DIR/data/raw'"
rsync -az --progress -e "$RSH" \
  "$ROOT"/data/raw/month_*.parquet "$REMOTE_HOST:$REMOTE_DIR/data/raw/"

echo "==> 2/3 installing uv + deps (CUDA torch) on the box"
$SSH "set -e; cd '$REMOTE_DIR'
  command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }
  command -v uv   >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  uv sync --extra dev
  uv run --extra dev python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'"

echo "==> 3/3 starting embed in remote tmux session 'embed'"
$SSH "cd '$REMOTE_DIR'; export PATH=\"\$HOME/.local/bin:\$PATH\"
  tmux kill-session -t embed 2>/dev/null || true
  tmux new-session -d -s embed \
    'uv run --extra dev python misc/generate_embeddings_gpu.py 2>&1 | tee embed.log'"

cat <<EOF

✅ Embedding running on the GPU box (tmux session 'embed').
   Watch : $RSH $REMOTE_HOST 'tmux attach -t embed'      (detach with Ctrl-b then d)
   Pull  : REMOTE_HOST=$REMOTE_HOST REMOTE_PORT=$REMOTE_PORT REMOTE_DIR=$REMOTE_DIR/data/embedded ./misc/sync_embeddings.sh
   Load  : make load        (on this laptop — it reaches Railway, not the GPU box)

Tip: destroy the vast.ai instance once data/embedded is fully synced back.
EOF
