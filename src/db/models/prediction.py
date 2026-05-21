"""Feature tables, model registry, predictions, drift events."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models.base import Base


class TeamFeature(Base):
    """Pre-computed feature vector for a team as-of a given timestamp."""
    __tablename__ = "team_features"

    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), primary_key=True)
    as_of_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class PlayerFeature(Base):
    """Pre-computed feature vector for a player as-of a given timestamp."""
    __tablename__ = "player_features"

    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True)
    as_of_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class MatchupFeature(Base):
    """Combined matchup-level feature vector for a specific game."""
    __tablename__ = "matchup_features"

    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ModelRecord(Base):
    """Mirror of MLflow model registry for SQL joins."""
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)    # 'winner', 'props'
    target: Mapped[str] = mapped_column(String(64), nullable=False)  # 'home_won', 'PTS', etc.
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    feature_spec_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    predictions: Mapped[list[Prediction]] = relationship("Prediction", back_populates="model")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("game_id", "model_id", "target", "player_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    target: Mapped[str] = mapped_column(String(64), nullable=False)  # 'home_won', 'PTS', etc.
    value: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))    # predicted mean/value
    probability: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # P(home_win) or P(over)
    quantiles: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # {"0.10": 12.3, ...}
    features_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    game: Mapped[Any] = relationship("Game", back_populates="predictions")
    model: Mapped[ModelRecord] = relationship("ModelRecord", back_populates="predictions")
    player: Mapped[Any | None] = relationship("Player")
    audit: Mapped[list[PredictionAudit]] = relationship("PredictionAudit", back_populates="prediction")


class PredictionAudit(Base):
    __tablename__ = "predictions_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False, index=True)
    raw_features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    prediction: Mapped[Prediction] = relationship("Prediction", back_populates="audit")


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)    # 'winner', 'props'
    target: Mapped[str] = mapped_column(String(64), nullable=False)
    drift_type: Mapped[str] = mapped_column(String(32), nullable=False)  # 'performance', 'calibration', 'psi', 'concept'
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    threshold: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False, default="normal")
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
