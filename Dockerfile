FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/app/.venv/bin:$PATH

# Expose port for gradio
EXPOSE 7860

# Default command
CMD ["python", "-m", "hn_search.rag.web_ui"]