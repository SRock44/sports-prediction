"""ORM models for game odds and weather data."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models.base import Base


class GameOdds(Base):
    __tablename__ = "game_odds"
    __table_args__ = (UniqueConstraint("game_id", "bookmaker", "market", "snapshot"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    bookmaker: Mapped[str] = mapped_column(String(32), nullable=False)  # 'draftkings', 'fanduel'
    market: Mapped[str] = mapped_column(String(16), nullable=False)     # 'h2h', 'spreads'
    snapshot: Mapped[str] = mapped_column(String(16), nullable=False)   # 'open', 'close'
    home_price: Mapped[float | None] = mapped_column(Float)   # American odds e.g. -110
    away_price: Mapped[float | None] = mapped_column(Float)
    home_spread: Mapped[float | None] = mapped_column(Float)  # spreads only e.g. -3.5
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    game: Mapped[Any] = relationship("Game")


class GameWeather(Base):
    __tablename__ = "game_weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, unique=True, index=True)
    temp_f: Mapped[float | None] = mapped_column(Float)
    wind_mph: Mapped[float | None] = mapped_column(Float)
    wind_bearing: Mapped[float | None] = mapped_column(Float)   # degrees, 0=N clockwise
    precip_prob: Mapped[float | None] = mapped_column(Float)    # 0.0-1.0
    conditions: Mapped[str | None] = mapped_column(String(64))  # 'clear','rain','cloudy'
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    game: Mapped[Any] = relationship("Game")
