"""Add game_odds and game_weather tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-22
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "game_odds",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("game_id", sa.Integer(), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("bookmaker", sa.String(32), nullable=False),
        sa.Column("market", sa.String(16), nullable=False),
        sa.Column("snapshot", sa.String(16), nullable=False),
        sa.Column("home_price", sa.Float(), nullable=True),
        sa.Column("away_price", sa.Float(), nullable=True),
        sa.Column("home_spread", sa.Float(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "bookmaker", "market", "snapshot"),
    )
    op.create_index("ix_game_odds_game_id", "game_odds", ["game_id"])

    op.create_table(
        "game_weather",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("game_id", sa.Integer(), sa.ForeignKey("games.id"), nullable=False, unique=True),
        sa.Column("temp_f", sa.Float(), nullable=True),
        sa.Column("wind_mph", sa.Float(), nullable=True),
        sa.Column("wind_bearing", sa.Float(), nullable=True),
        sa.Column("precip_prob", sa.Float(), nullable=True),
        sa.Column("conditions", sa.String(64), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_game_weather_game_id", "game_weather", ["game_id"])


def downgrade() -> None:
    op.drop_index("ix_game_weather_game_id", "game_weather")
    op.drop_table("game_weather")
    op.drop_index("ix_game_odds_game_id", "game_odds")
    op.drop_table("game_odds")
