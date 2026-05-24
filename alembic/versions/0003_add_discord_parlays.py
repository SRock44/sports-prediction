"""Add discord_parlays table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-23
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "discord_parlays",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("discord_user_id", sa.String(length=64), nullable=False),
        sa.Column("discord_username", sa.String(length=128), nullable=False),
        sa.Column("discord_message_id", sa.String(length=64), nullable=True),
        sa.Column("discord_channel_id", sa.String(length=64), nullable=True),
        sa.Column("sport_code", sa.String(length=8), nullable=False),
        sa.Column("bookmaker", sa.String(length=32), nullable=False),
        sa.Column("n_legs", sa.Integer(), nullable=False),
        sa.Column(
            "legs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("parlay_odds_american", sa.Integer(), nullable=True),
        sa.Column("parlay_ev", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("n_correct", sa.Integer(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_discord_parlays_status", "discord_parlays", ["status"])
    op.create_index("ix_discord_parlays_sport_code", "discord_parlays", ["sport_code"])
    op.create_index("ix_discord_parlays_created_at", "discord_parlays", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_discord_parlays_created_at", table_name="discord_parlays")
    op.drop_index("ix_discord_parlays_sport_code", table_name="discord_parlays")
    op.drop_index("ix_discord_parlays_status", table_name="discord_parlays")
    op.drop_table("discord_parlays")
