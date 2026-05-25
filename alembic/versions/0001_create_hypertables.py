"""Convert time-series tables to TimescaleDB hypertables.

Revision ID: 0001_create_hypertables
Revises:
Create Date: 2026-05-21

This migration must run AFTER the initial schema creation (Alembic autogenerate
or Base.metadata.create_all). It calls create_hypertable on the three tables
that are designed for time-series workloads. The operation is a no-op if
TimescaleDB is not installed (e.g. plain Postgres in CI).
"""

from __future__ import annotations

from alembic import op

revision = "0001_create_hypertables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Check TimescaleDB is available; skip gracefully in plain-Postgres environments.
    result = conn.execute("SELECT count(*) FROM pg_extension WHERE extname = 'timescaledb'")
    if result.scalar() == 0:
        return

    for table, time_col in (
        ("plays", "occurred_at"),
        ("team_game_stats", "recorded_at"),
        ("player_game_stats", "recorded_at"),
    ):
        conn.execute(
            f"SELECT create_hypertable('{table}', '{time_col}', "
            f"if_not_exists => TRUE, migrate_data => TRUE)"
        )


def downgrade() -> None:
    # Hypertables cannot be trivially reverted without data loss; document only.
    pass
