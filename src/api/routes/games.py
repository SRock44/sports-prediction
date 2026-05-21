"""GET /v1/games/upcoming, GET /v1/games/{game_id}"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import require_scope, limiter
from src.core.time import utc_now
from src.db.session import get_async_session
from src.db.repositories.games import get_upcoming_games, get_game_by_id

router = APIRouter(tags=["games"])


@router.get("/games/upcoming")
async def upcoming_games(
    sport: str = Query(..., pattern="^(nba|mlb)$"),
    hours: int = Query(default=48, ge=1, le=168),
    payload: dict = Depends(require_scope("predictions:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[dict[str, Any]]:
    now = utc_now()
    games = await get_upcoming_games(session, sport, now, now + timedelta(hours=hours))
    return [_game_to_dict(g) for g in games]


@router.get("/games/{game_id}")
async def game_detail(
    game_id: int,
    payload: dict = Depends(require_scope("predictions:read")),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    game = await get_game_by_id(session, game_id)
    if game is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Game not found")
    return _game_to_dict(game)


def _game_to_dict(g: Any) -> dict[str, Any]:
    return {
        "id": g.id,
        "sport": g.sport_id,
        "external_id": g.external_id,
        "season": g.season,
        "scheduled_utc": g.scheduled_utc.isoformat(),
        "status": g.status,
        "home_team": {"id": g.home_team.id, "name": g.home_team.name, "abbrev": g.home_team.abbrev},
        "away_team": {"id": g.away_team.id, "name": g.away_team.name, "abbrev": g.away_team.abbrev},
        "venue": {"name": g.venue.name, "city": g.venue.city} if g.venue else None,
        "home_score": g.home_score,
        "away_score": g.away_score,
    }
