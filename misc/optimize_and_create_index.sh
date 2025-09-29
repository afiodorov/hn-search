#!/bin/bash

echo "Optimizing PostgreSQL settings for HNSW index creation..."
echo ""

# First, cancel any existing index creation
echo "Canceling any existing index creation..."
docker exec hn-search-postgres-1 psql -U postgres -d hn_search -c \
  "SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE query LIKE '%CREATE INDEX%hn_documents_embedding_idx%' AND state = 'active';"

# Increase maintenance_work_mem for this session (4GB for better performance)
echo "Setting maintenance_work_mem to 4GB for faster index creation..."
docker exec hn-search-postgres-1 psql -U postgres -d hn_search -c \
  "SET maintenance_work_mem = '4GB'; CREATE INDEX IF NOT EXISTS hn_documents_embedding_idx ON hn_documents USING hnsw (embedding vector_cosine_ops);" &

PID=$!
echo ""
echo "Index creation started with PID: $PID"
echo "Using 4GB of maintenance memory for faster building"
echo ""
echo "Note: The warning about memory after 16,758 tuples is expected for large datasets."
echo "With 4GB memory, it should handle much more before needing to spill to disk."
echo ""
echo "You can check progress with:"
echo "  docker exec hn-search-postgres-1 psql -U postgres -d hn_search -c \"SELECT * FROM pg_stat_progress_create_index;\""
echo ""
echo "Or check active queries:"
echo "  docker exec hn-search-postgres-1 psql -U postgres -d hn_search -c \"SELECT pid, now() - query_start AS duration, query FROM pg_stat_activity WHERE state = 'active' AND query LIKE '%INDEX%';\""