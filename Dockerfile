# syntax=docker/dockerfile:1
# Multi-stage: builder installs deps; final is lean and rootless.
# The same image is used for api, worker, beat, and notifier containers —
# the CMD is overridden per-service in docker-compose.yml.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps only (not carried into final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip deps into a prefix we can copy
# src/ must be present so the package metadata can be resolved during install.
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir ".[dev]"

# ── Stage 2: final ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Ensure non-root can write to writable paths only
RUN mkdir -p mlruns reports /tmp/pybaseball_cache && \
    chown -R app:app /app mlruns reports /tmp/pybaseball_cache

USER app

# Pybaseball writes a cache; redirect it to /tmp which is writable
ENV PYBASEBALL_CACHE=/tmp/pybaseball_cache

# The default CMD runs the API. Override in docker-compose per service.
CMD ["gunicorn", "src.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1
