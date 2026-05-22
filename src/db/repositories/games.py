"""Query helpers for games and stats. No business logic here."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models.sport import Game, Sport


async def get_upcoming_games(
    session: AsyncSession,
    sport_code: str,
    after_utc: datetime,
    before_utc: datetime,
) -> list[Game]:
    result = await session.execute(
        select(Game)
        .join(Sport, Game.sport_id == Sport.id)
        .where(
            Sport.code == sport_code,
            Game.scheduled_utc >= after_utc,
            Game.scheduled_utc <= before_utc,
            Game.status.in_(["scheduled", "pre-game"]),
        )
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
            selectinload(Game.venue),
        )
        .order_by(Game.scheduled_utc)
    )
    return list(result.scalars().all())


async def get_game_by_id(session: AsyncSession, game_id: int) -> Game | None:
    result = await session.execute(
        select(Game)
        .where(Game.id == game_id)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
            selectinload(Game.venue),
        )
    )
    return result.scalar_one_or_none()


async def get_games_for_season(
    session: AsyncSession,
    sport_code: str,
    season: int,
) -> list[Game]:
    result = await session.execute(
        select(Game)
        .join(Sport, Game.sport_id == Sport.id)
        .where(Sport.code == sport_code, Game.season == season)
        .order_by(Game.scheduled_utc)
    )
    return list(result.scalars().all())
