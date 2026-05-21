"""GET /v1/predictions/game/{id}, /v1/predictions/props/{id}, /v1/predictions/player/{id}"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import require_scope
from src.db.session import get_async_session
from src.db.repositories.predictions import (
    get_predictions_for_game,
    get_player_prediction,
)

router = APIRouter(tags=["predictions"])


@router.get("/predictions/game/{game_id}")
async def game_prediction(
    game_id: int,
    payload: dict = Depends(require_scope("predictions:read")),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Win/loss prediction for a game."""
    preds = await get_predictions_for_game(session, game_id)
    winner_preds = [p for p in preds if p.target == "home_won" and p.player_id is None]

    if not winner_preds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No winner prediction found")

    p = winner_preds[0]
    return {
        "game_id": game_id,
        "target": "home_won",
        "home_win_probability": float(p.probability or 0),
        "away_win_probability": round(1.0 - float(p.probability or 0), 4),
        "model_version": p.model.version,
        "as_of_utc": p.created_at.isoformat(),
        "features_hash": p.features_hash,
    }


@router.get("/predictions/props/{game_id}")
async def game_props(
    game_id: int,
    payload: dict = Depends(require_scope("predictions:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[dict[str, Any]]:
    """All player prop predictions for a game."""
    preds = await get_predictions_for_game(session, game_id)
    prop_preds = [p for p in preds if p.target != "home_won" and p.player_id is not None]

    return [
        {
            "player_id": p.player_id,
            "player_name": p.player.full_name if p.player else None,
            "target": p.target,
            "predicted_median": float(p.value or 0),
            "quantiles": p.quantiles,
            "model_version": p.model.version,
            "as_of_utc": p.created_at.isoformat(),
            "features_hash": p.features_hash,
        }
        for p in prop_preds
    ]


@router.get("/predictions/player/{player_id}")
async def player_props(
    player_id: int,
    game_id: int = Query(...),
    payload: dict = Depends(require_scope("predictions:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[dict[str, Any]]:
    """Props for a specific player in a specific game."""
    preds = await get_player_prediction(session, game_id, player_id)
    if not preds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No props found")

    return [
        {
            "player_id": player_id,
            "target": p.target,
            "predicted_median": float(p.value or 0),
            "quantiles": p.quantiles,
            "model_version": p.model.version,
            "as_of_utc": p.created_at.isoformat(),
        }
        for p in preds
    ]
