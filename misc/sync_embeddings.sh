#!/bin/bash

REMOTE_HOST="root@86.127.249.120"
REMOTE_PORT="21604"
REMOTE_DIR="/hn/embeddings"
LOCAL_DIR="/Users/artiomfiodorov/code/hn-search/embeddings"

mkdir -p "$LOCAL_DIR"

while true; do
    echo "$(date): Checking for completed files..."

    completed_files=$(ssh -p "$REMOTE_PORT" "$REMOTE_HOST" '
        cd '"$REMOTE_DIR"' 2>/dev/null || exit 0
        for f in *.parquet; do
            [ -f "$f" ] || continue
            size1=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            sleep 2
            size2=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            if [ "$size1" = "$size2" ]; then
                echo "$f"
            fi
        done
    ')

    if [ -n "$completed_files" ]; then
        echo "$(date): Found completed files, syncing..."
        for file in $completed_files; do
            rsync -avz --remove-source-files -e "ssh -p $REMOTE_PORT" \
                "$REMOTE_HOST:$REMOTE_DIR/$file" "$LOCAL_DIR/"
            echo "$(date): Synced $file"
        done
    else
        echo "$(date): No completed files to sync"
    fi

    sleep 60
done
