"""Query helpers for predictions and model records."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models.prediction import ModelRecord, Prediction


async def get_active_model(
    session: AsyncSession, sport_code: str, kind: str, target: str
) -> ModelRecord | None:
    from src.db.models.sport import Sport

    result = await session.execute(
        select(ModelRecord)
        .join(Sport, ModelRecord.sport_id == Sport.id)
        .where(
            Sport.code == sport_code,
            ModelRecord.kind == kind,
            ModelRecord.target == target,
            ModelRecord.active.is_(True),
        )
        .order_by(ModelRecord.trained_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_predictions_for_game(session: AsyncSession, game_id: int) -> list[Prediction]:
    result = await session.execute(
        select(Prediction)
        .where(Prediction.game_id == game_id)
        .options(selectinload(Prediction.model), selectinload(Prediction.player))
        .order_by(Prediction.target)
    )
    return list(result.scalars().all())


async def get_player_prediction(
    session: AsyncSession, game_id: int, player_id: int
) -> list[Prediction]:
    result = await session.execute(
        select(Prediction)
        .where(Prediction.game_id == game_id, Prediction.player_id == player_id)
        .options(selectinload(Prediction.model))
    )
    return list(result.scalars().all())
