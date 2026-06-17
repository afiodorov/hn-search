#!/usr/bin/env bash
# Ship locally-built artifacts to the VPS into a fresh release dir, then atomically
# flip the `current` symlink and restart the service, so live queries never read a
# half-written file.
#
# Usage:  REMOTE=hnsearch@vps ./scripts/rsync_artifacts.sh [LOCAL_DIR]
#
# A fresh base release starts with an EMPTY tail. That is safe: the service reports
# its new max_id (the dump's high-water mark) and the next daily updater run re-fetches
# id > max_id from BigQuery, re-appending anything added since the dump. So only flip
# to a release built from a complete, up-to-date dump (or during a maintenance window).
set -euo pipefail

LOCAL_DIR="${1:-artifacts}"
REMOTE="${REMOTE:?set REMOTE=user@host}"
REMOTE_BASE="${REMOTE_BASE:-/var/lib/hnsearch}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$REMOTE_BASE/releases/$STAMP"

for f in codes.bin rerank_f16.bin docs.sqlite meta.json; do
	[[ -f "$LOCAL_DIR/$f" ]] || { echo "missing $LOCAL_DIR/$f" >&2; exit 1; }
done

echo "rsync $LOCAL_DIR -> $REMOTE:$DEST"
ssh "$REMOTE" "mkdir -p '$DEST'"
# --partial/--inplace make re-runs resumable and send only changed (appended) bytes.
rsync -avP --partial --inplace \
	"$LOCAL_DIR/codes.bin" "$LOCAL_DIR/rerank_f16.bin" \
	"$LOCAL_DIR/docs.sqlite" "$LOCAL_DIR/meta.json" \
	"$REMOTE:$DEST/"

echo "flip current -> $DEST and restart"
ssh "$REMOTE" "ln -sfn '$DEST' '$REMOTE_BASE/current' && sudo systemctl restart hnsearch"
echo "done. Keep the previous release dir around for instant rollback."
