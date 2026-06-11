import os

import uvicorn

from hn_search.logging_config import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🔎 Starting HN RAG Search API on port {port}...")
    uvicorn.run("hn_search.api.app:app", host="0.0.0.0", port=port)
