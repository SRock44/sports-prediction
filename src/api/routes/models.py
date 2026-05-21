"""GET /v1/models/active, GET /v1/sports"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import require_scope
from src.db.session import get_async_session
from src.db.models.prediction import ModelRecord
from src.db.models.sport import Sport

router = APIRouter(tags=["models"])


@router.get("/sports")
async def list_sports(
    payload: dict = Depends(require_scope("predictions:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[dict[str, Any]]:
    result = await session.execute(select(Sport))
    sports = result.scalars().all()
    return [{"id": s.id, "code": s.code} for s in sports]


@router.get("/models/active")
async def active_models(
    sport: str | None = Query(default=None, pattern="^(nba|mlb)$"),
    payload: dict = Depends(require_scope("models:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[dict[str, Any]]:
    """Return active model records, optionally filtered by sport."""
    stmt = select(ModelRecord).where(ModelRecord.active.is_(True))
    if sport:
        result_sport = await session.execute(select(Sport).where(Sport.code == sport))
        sport_obj = result_sport.scalar_one_or_none()
        if sport_obj:
            stmt = stmt.where(ModelRecord.sport_id == sport_obj.id)

    result = await session.execute(stmt.order_by(ModelRecord.trained_at.desc()))
    models = result.scalars().all()

    return [
        {
            "id": m.id,
            "sport_id": m.sport_id,
            "kind": m.kind,
            "target": m.target,
            "version": m.version,
            "trained_at": m.trained_at.isoformat(),
            "metrics": m.metrics,
        }
        for m in models
    ]
