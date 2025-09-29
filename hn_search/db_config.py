import os
from urllib.parse import urlparse


def get_db_config():
    """
    Get database configuration from environment variables.
    Supports both Railway environment variables and DATABASE_URL.
    """
    # First try DATABASE_URL (Railway standard)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        parsed = urlparse(database_url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "dbname": parsed.path[1:] if parsed.path else "postgres",
            "user": parsed.username,
            "password": parsed.password,
        }

    # Fallback to individual Railway environment variables
    return {
        "host": os.getenv("PGHOST", os.getenv("PGHOST_PRIVATE", "localhost")),
        "port": int(os.getenv("PGPORT", os.getenv("PGPORT_PRIVATE", "5432"))),
        "dbname": os.getenv("PGDATABASE", os.getenv("POSTGRES_DB", "hn_search")),
        "user": os.getenv("PGUSER", os.getenv("POSTGRES_USER", "postgres")),
        "password": os.getenv("PGPASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres")),
    }
