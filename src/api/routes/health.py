"""GET /v1/health — liveness + readiness check."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from src.core.time import utc_now

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, Any]:
    from src.db.session import async_session_factory
    from src.core.config import settings

    checks: dict[str, Any] = {"status": "ok", "checks": {}}

    # ── Postgres ──────────────────────────────────────────────────────────────
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["checks"]["postgres"] = "ok"
    except Exception as exc:
        checks["checks"]["postgres"] = f"ERROR: {exc}"
        checks["status"] = "degraded"

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        import redis.asyncio as aioredis
        r = await aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        checks["checks"]["redis"] = "ok"
    except Exception as exc:
        checks["checks"]["redis"] = f"ERROR: {exc}"
        checks["status"] = "degraded"

    # ── Last ingest freshness ─────────────────────────────────────────────────
    # NBA regular season: Oct–Jun. MLB: Apr–Oct. Outside those windows the
    # freshness check is skipped and reported as "offseason".
    _NBA_IN_SEASON_MONTHS = frozenset(range(1, 7)) | frozenset(range(10, 13))  # Oct-Jun
    _MLB_IN_SEASON_MONTHS = frozenset(range(4, 11))                             # Apr-Oct
    _SEASON_MONTHS = {"nba": _NBA_IN_SEASON_MONTHS, "mlb": _MLB_IN_SEASON_MONTHS}

    try:
        async with async_session_factory() as session:
            now = utc_now()
            for sport in ("nba", "mlb"):
                if now.month not in _SEASON_MONTHS[sport]:
                    checks["checks"][f"{sport}_ingest_fresh"] = "offseason"
                    continue

                row = await session.execute(
                    text("""
                        SELECT MAX(g.scheduled_utc) as last_game
                        FROM games g JOIN sports s ON s.id=g.sport_id
                        WHERE s.code = :code AND g.status='final'
                    """),
                    {"code": sport},
                )
                last = row.scalar()
                if last is not None:
                    age_hours = (now - last).total_seconds() / 3600
                    checks["checks"][f"{sport}_ingest_age_hours"] = round(age_hours, 1)
                    if age_hours > 25:
                        checks["status"] = "degraded"
                        checks["checks"][f"{sport}_ingest_fresh"] = False
                    else:
                        checks["checks"][f"{sport}_ingest_fresh"] = True
                else:
                    checks["status"] = "degraded"
                    checks["checks"][f"{sport}_ingest_fresh"] = False
    except Exception as exc:
        checks["checks"]["ingest_check"] = f"ERROR: {exc}"

    # ── Active model freshness ────────────────────────────────────────────────
    try:
        async with async_session_factory() as session:
            row = await session.execute(
                text("SELECT MAX(trained_at) FROM models WHERE active=true")
            )
            last_trained = row.scalar()
            if last_trained is not None:
                age_days = (utc_now() - last_trained).total_seconds() / 86400
                checks["checks"]["model_age_days"] = round(age_days, 1)
                if age_days > 14:
                    checks["checks"]["model_fresh"] = False
    except Exception as exc:
        checks["checks"]["model_check"] = f"ERROR: {exc}"

    checks["ts"] = utc_now().isoformat()
    return checks
