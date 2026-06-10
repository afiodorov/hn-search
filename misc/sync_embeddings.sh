#!/bin/bash
# Poll a remote GPU box (e.g. vast.ai) and rsync finished embedded month files back
# as they complete, so you can pull results while the rest keep computing.
#
# Configure via env vars (vast.ai gives you a fresh host/port each rental):
#   REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22 REMOTE_DIR=/root/hn-search/data/embedded \
#     ./misc/sync_embeddings.sh
#
# Files are considered "done" when their size is stable across a 2s check, then
# rsynced into ./data/embedded (use --remove-source-files only if you're sure).

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST, e.g. root@1.2.3.4}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/root/hn-search/data/embedded}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/embedded}"

mkdir -p "$LOCAL_DIR"
echo "Syncing $REMOTE_HOST:$REMOTE_DIR -> $LOCAL_DIR (Ctrl-C to stop)"

while true; do
    echo "$(date): checking for completed files..."
    completed_files=$(ssh -p "$REMOTE_PORT" "$REMOTE_HOST" '
        cd '"$REMOTE_DIR"' 2>/dev/null || exit 0
        for f in month_*.parquet; do
            [ -f "$f" ] || continue
            s1=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            sleep 2
            s2=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            [ "$s1" = "$s2" ] && echo "$f"
        done
    ')

    if [ -n "$completed_files" ]; then
        for file in $completed_files; do
            rsync -avz -e "ssh -p $REMOTE_PORT" \
                "$REMOTE_HOST:$REMOTE_DIR/$file" "$LOCAL_DIR/"
            echo "$(date): synced $file"
        done
    else
        echo "$(date): nothing new"
    fi
    sleep 60
done
