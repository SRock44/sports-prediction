"""Core sports-domain entities: Sport, Team, Player, Venue."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models.base import Base


class Sport(Base):
    __tablename__ = "sports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)  # 'nba', 'mlb'

    teams: Mapped[list[Team]] = relationship("Team", back_populates="sport")
    players: Mapped[list[Player]] = relationship("Player", back_populates="sport")
    venues: Mapped[list[Venue]] = relationship("Venue", back_populates="sport")
    games: Mapped[list[Game]] = relationship("Game", back_populates="sport")


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("sport_id", "external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    abbrev: Mapped[str] = mapped_column(String(10), nullable=False)
    conference: Mapped[str | None] = mapped_column(String(64))
    division: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    sport: Mapped[Sport] = relationship("Sport", back_populates="teams")
    home_games: Mapped[list[Game]] = relationship(
        "Game", foreign_keys="Game.home_team_id", back_populates="home_team"
    )
    away_games: Mapped[list[Game]] = relationship(
        "Game", foreign_keys="Game.away_team_id", back_populates="away_team"
    )


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("sport_id", "external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    primary_position: Mapped[str | None] = mapped_column(String(32))
    bats: Mapped[str | None] = mapped_column(String(1))  # MLB: L/R/S
    throws: Mapped[str | None] = mapped_column(String(1))  # MLB: L/R
    birthdate: Mapped[date | None] = mapped_column(Date)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    sport: Mapped[Sport] = relationship("Sport", back_populates="players")
    injuries: Mapped[list[Injury]] = relationship("Injury", back_populates="player")


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    city: Mapped[str | None] = mapped_column(String(64))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    indoor: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    sport: Mapped[Sport] = relationship("Sport", back_populates="venues")
    games: Mapped[list[Game]] = relationship("Game", back_populates="venue")


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (UniqueConstraint("sport_id", "external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    scheduled_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="scheduled")
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    sport: Mapped[Sport] = relationship("Sport", back_populates="games")
    home_team: Mapped[Team] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_games"
    )
    away_team: Mapped[Team] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_games"
    )
    venue: Mapped[Venue | None] = relationship("Venue", back_populates="games")
    team_stats: Mapped[list[TeamGameStats]] = relationship("TeamGameStats", back_populates="game")
    player_stats: Mapped[list[PlayerGameStats]] = relationship(
        "PlayerGameStats", back_populates="game"
    )
    plays: Mapped[list[Play]] = relationship("Play", back_populates="game")
    lineups: Mapped[list[Lineup]] = relationship("Lineup", back_populates="game")
    predictions: Mapped[list[Prediction]] = relationship("Prediction", back_populates="game")

    @property
    def home_won(self) -> bool | None:
        if self.home_score is None or self.away_score is None:
            return None
        return self.home_score > self.away_score


class TeamGameStats(Base):
    __tablename__ = "team_game_stats"

    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), primary_key=True)
    # recorded_at is required as the TimescaleDB hypertable time dimension.
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    game: Mapped[Game] = relationship("Game", back_populates="team_stats")
    team: Mapped[Team] = relationship("Team")


class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"

    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    # recorded_at is required as the TimescaleDB hypertable time dimension.
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    game: Mapped[Game] = relationship("Game", back_populates="player_stats")
    player: Mapped[Player] = relationship("Player")
    team: Mapped[Team] = relationship("Team")


class Play(Base):
    """Individual play-by-play event. Intended as a TimescaleDB hypertable on occurred_at."""

    __tablename__ = "plays"
    __table_args__ = (UniqueConstraint("game_id", "action_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    # occurred_at is the TimescaleDB hypertable time dimension.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    action_number: Mapped[int] = mapped_column(Integer, nullable=False)
    period: Mapped[int] = mapped_column(Integer, nullable=False)
    clock: Mapped[str | None] = mapped_column(String(16))  # "PT08M32.00S"
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    sub_type: Mapped[str | None] = mapped_column(String(64))
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    score_home: Mapped[int | None] = mapped_column(Integer)
    score_away: Mapped[int | None] = mapped_column(Integer)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    game: Mapped[Game] = relationship("Game", back_populates="plays")


class Lineup(Base):
    __tablename__ = "lineups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # 'official', 'projected'
    players: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    game: Mapped[Game] = relationship("Game", back_populates="lineups")


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # 'out', 'questionable', 'dtd', etc.
    reason: Mapped[str | None] = mapped_column(Text)
    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    expected_return_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    player: Mapped[Player] = relationship("Player", back_populates="injuries")


# Import at bottom to avoid circular reference
from src.db.models.prediction import Prediction  # noqa: E402
