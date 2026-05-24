"""Discord parlay tracking — records every parlay a user locks in and its outcome."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.db.models.base import Base


class DiscordParlay(Base):
    """One parlay built by a Discord user via the /predict command."""

    __tablename__ = "discord_parlays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Discord identity
    discord_user_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    discord_username: Mapped[str] = mapped_column(String(100), nullable=False)
    discord_message_id: Mapped[str | None] = mapped_column(String(32), index=True)
    discord_channel_id: Mapped[str | None] = mapped_column(String(32))

    # Parlay config
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False)  # 'nba' | 'mlb'
    bookmaker: Mapped[str] = mapped_column(String(32), nullable=False)  # 'draftkings' | 'fanduel'
    n_legs: Mapped[int] = mapped_column(Integer, nullable=False)

    # Legs: list of {game_id, home_team, away_team, pick, model_prob,
    #                implied_prob, odds_american, scheduled_utc}
    legs: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    # Combined odds + EV
    parlay_odds_american: Mapped[int | None] = mapped_column(Integer)
    parlay_ev: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))

    # Outcome tracking
    # 'pending' | 'won' | 'lost' | 'push' | 'partial'
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    n_correct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
