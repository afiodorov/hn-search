FROM python:3.13-slim

WORKDIR /app

# install uv via official installer (specific known-good version)
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /usr/local/bin/uv

# copy lockfile + pyproject first so the layer is cached unless deps change
COPY pyproject.toml uv.lock .python-version ./

# install only serve deps (no torch/sentence-transformers)
RUN uv sync --frozen --no-dev --no-install-project

# copy app code
COPY hn_search/ hn_search/

# uv puts the venv at .venv; activate it so the CMD can use plain `python`
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
CMD ["python", "-m", "hn_search.rag.web_ui"]
