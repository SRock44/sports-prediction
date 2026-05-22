"""Re-export all ORM models so Alembic autogenerate picks them up."""

from src.db.models.auth import ApiKey, ApiRequest
from src.db.models.base import Base
from src.db.models.odds import GameOdds, GameWeather
from src.db.models.prediction import (
    DriftEvent,
    MatchupFeature,
    ModelRecord,
    PlayerFeature,
    Prediction,
    PredictionAudit,
    TeamFeature,
)
from src.db.models.sport import (
    Game,
    Injury,
    Lineup,
    Play,
    Player,
    PlayerGameStats,
    Sport,
    Team,
    TeamGameStats,
    Venue,
)

__all__ = [
    "ApiKey",
    "ApiRequest",
    "Base",
    "DriftEvent",
    "Game",
    "GameOdds",
    "GameWeather",
    "Injury",
    "Lineup",
    "MatchupFeature",
    "ModelRecord",
    "Play",
    "Player",
    "PlayerFeature",
    "PlayerGameStats",
    "Prediction",
    "PredictionAudit",
    "Sport",
    "Team",
    "TeamFeature",
    "TeamGameStats",
    "Venue",
]
