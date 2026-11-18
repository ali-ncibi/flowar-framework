# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

# Install uv (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application source and install the project itself (editable)
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" || exit 1

CMD ["uv", "run", "--no-sync", "streamlit", "run", "frontend/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
