"""FastAPI application factory."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_prometheus import PrometheusMiddleware, metrics as prometheus_metrics

from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.api.routes import games, predictions, models, health
from src.api import auth

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info("api.startup", environment=settings.environment)
    yield
    log.info("api.shutdown")


app = FastAPI(
    title="Sports Prediction API",
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

# CORS — deny by default; only configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Prometheus metrics (before auth middleware so /metrics is accessible)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", prometheus_metrics)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        response.headers.pop("Server", None)
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Async audit log: writes to api_requests table for every authenticated call."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = int((time.monotonic() - start) * 1000)

        key_id: int | None = getattr(request.state, "api_key_id", None)
        if key_id is not None:
            import asyncio
            asyncio.create_task(
                _write_audit(
                    key_id=key_id,
                    route=request.url.path,
                    status=response.status_code,
                    latency_ms=latency_ms,
                    ip=request.client.host if request.client else None,
                )
            )
        return response


async def _write_audit(
    key_id: int, route: str, status: int, latency_ms: int, ip: str | None
) -> None:
    from src.core.time import utc_now
    from src.db.session import async_session_factory
    from src.db.models.auth import ApiRequest

    async with async_session_factory() as session:
        session.add(ApiRequest(
            api_key_id=key_id,
            route=route,
            status=status,
            latency_ms=latency_ms,
            ip=ip,
            ts=utc_now(),
        ))
        await session.commit()


app.add_middleware(AuditLogMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Import Any for type annotation
from typing import Any  # noqa: E402

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/v1")
app.include_router(auth.router, prefix="/v1")
app.include_router(games.router, prefix="/v1")
app.include_router(predictions.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")
