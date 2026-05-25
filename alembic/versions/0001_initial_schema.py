"""Initial schema: all tables and TimescaleDB hypertables.

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extensions ────────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # ── sports ────────────────────────────────────────────────────────────────
    op.create_table(
        "sports",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(20), nullable=False, unique=True),
    )

    # ── teams ─────────────────────────────────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "sport_id", sa.Integer, sa.ForeignKey("sports.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("abbrev", sa.String(10)),
        sa.Column("conference", sa.String(64)),
        sa.Column("division", sa.String(64)),
        sa.Column("meta", postgresql.JSONB),
        sa.UniqueConstraint("sport_id", "external_id", name="uq_teams_sport_external"),
    )

    # ── players ───────────────────────────────────────────────────────────────
    op.create_table(
        "players",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "sport_id", sa.Integer, sa.ForeignKey("sports.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("full_name", sa.String(256), nullable=False),
        sa.Column("primary_position", sa.String(20)),
        sa.Column("bats", sa.String(5)),
        sa.Column("throws", sa.String(5)),
        sa.Column("birthdate", sa.Date),
        sa.Column("meta", postgresql.JSONB),
        sa.UniqueConstraint("sport_id", "external_id", name="uq_players_sport_external"),
    )
    op.create_index(
        "ix_players_full_name_trgm",
        "players",
        ["full_name"],
        postgresql_using="gin",
        postgresql_ops={"full_name": "gin_trgm_ops"},
    )

    # ── venues ────────────────────────────────────────────────────────────────
    op.create_table(
        "venues",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "sport_id", sa.Integer, sa.ForeignKey("sports.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("city", sa.String(128)),
        sa.Column("lat", sa.Numeric(9, 6)),
        sa.Column("lon", sa.Numeric(9, 6)),
        sa.Column("indoor", sa.Boolean, default=True),
        sa.Column("meta", postgresql.JSONB),
    )

    # ── games ─────────────────────────────────────────────────────────────────
    op.create_table(
        "games",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "sport_id", sa.Integer, sa.ForeignKey("sports.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("season", sa.Integer),
        sa.Column("scheduled_utc", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(32), server_default="scheduled"),
        sa.Column("home_team_id", sa.Integer, sa.ForeignKey("teams.id")),
        sa.Column("away_team_id", sa.Integer, sa.ForeignKey("teams.id")),
        sa.Column("venue_id", sa.Integer, sa.ForeignKey("venues.id")),
        sa.Column("home_score", sa.Integer),
        sa.Column("away_score", sa.Integer),
        sa.Column("meta", postgresql.JSONB),
        sa.UniqueConstraint("sport_id", "external_id", name="uq_games_sport_external"),
    )
    op.create_index("ix_games_scheduled_utc", "games", ["scheduled_utc"])
    op.create_index("ix_games_status", "games", ["status"])

    # ── team_game_stats (hypertable) ──────────────────────────────────────────
    op.create_table(
        "team_game_stats",
        sa.Column(
            "game_id", sa.Integer, sa.ForeignKey("games.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("stats", postgresql.JSONB),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("game_id", "team_id"),
    )
    op.execute("SELECT create_hypertable('team_game_stats', 'recorded_at', if_not_exists => TRUE);")

    # ── player_game_stats (hypertable) ────────────────────────────────────────
    op.create_table(
        "player_game_stats",
        sa.Column(
            "game_id", sa.Integer, sa.ForeignKey("games.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "player_id", sa.Integer, sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id")),
        sa.Column("stats", postgresql.JSONB),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("game_id", "player_id"),
    )
    op.execute(
        "SELECT create_hypertable('player_game_stats', 'recorded_at', if_not_exists => TRUE);"
    )

    # ── plays (hypertable) ────────────────────────────────────────────────────
    op.create_table(
        "plays",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "game_id", sa.Integer, sa.ForeignKey("games.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("sequence", sa.Integer),
        sa.Column("period", sa.Integer),
        sa.Column("clock", sa.String(16)),
        sa.Column("event_type", sa.String(64)),
        sa.Column("payload", postgresql.JSONB),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.execute("SELECT create_hypertable('plays', 'recorded_at', if_not_exists => TRUE);")

    # ── lineups ───────────────────────────────────────────────────────────────
    op.create_table(
        "lineups",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "game_id", sa.Integer, sa.ForeignKey("games.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id")),
        sa.Column("source", sa.String(64)),
        sa.Column("players", postgresql.JSONB),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
    )

    # ── injuries ──────────────────────────────────────────────────────────────
    op.create_table(
        "injuries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "player_id", sa.Integer, sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_return_date", sa.Date),
        sa.Column("source", sa.String(64)),
    )
    op.create_index("ix_injuries_player_reported", "injuries", ["player_id", "reported_at"])

    # ── team_features ─────────────────────────────────────────────────────────
    op.create_table(
        "team_features",
        sa.Column(
            "team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("as_of_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sport_id", sa.Integer, sa.ForeignKey("sports.id")),
        sa.Column("features", postgresql.JSONB),
        sa.PrimaryKeyConstraint("team_id", "as_of_utc"),
    )

    # ── player_features ───────────────────────────────────────────────────────
    op.create_table(
        "player_features",
        sa.Column(
            "player_id", sa.Integer, sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("as_of_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("features", postgresql.JSONB),
        sa.PrimaryKeyConstraint("player_id", "as_of_utc"),
    )

    # ── matchup_features ──────────────────────────────────────────────────────
    op.create_table(
        "matchup_features",
        sa.Column(
            "game_id", sa.Integer, sa.ForeignKey("games.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("features", postgresql.JSONB),
        sa.Column("computed_at", sa.DateTime(timezone=True)),
    )

    # ── models ────────────────────────────────────────────────────────────────
    op.create_table(
        "models",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("sport_id", sa.Integer, sa.ForeignKey("sports.id"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("target", sa.String(64), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("mlflow_run_id", sa.String(128)),
        sa.Column("trained_at", sa.DateTime(timezone=True)),
        sa.Column("active", sa.Boolean, server_default="false"),
        sa.Column("metrics", postgresql.JSONB),
        sa.Column("feature_spec_hash", sa.String(64)),
    )
    op.create_index("ix_models_active", "models", ["sport_id", "kind", "active"])

    # ── predictions ───────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "game_id", sa.Integer, sa.ForeignKey("games.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("model_id", sa.Integer, sa.ForeignKey("models.id"), nullable=False),
        sa.Column("player_id", sa.Integer, sa.ForeignKey("players.id")),
        sa.Column("target", sa.String(64), nullable=False),
        sa.Column("value", sa.Numeric(10, 4)),
        sa.Column("probability", sa.Numeric(6, 4)),
        sa.Column("quantiles", postgresql.JSONB),
        sa.Column("features_hash", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "game_id",
            "model_id",
            "target",
            name="uq_predictions_game_model_target",
        ),
    )
    op.create_index("ix_predictions_game_id", "predictions", ["game_id"])

    # ── predictions_audit ─────────────────────────────────────────────────────
    op.create_table(
        "predictions_audit",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "prediction_id",
            sa.Integer,
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("raw_features", postgresql.JSONB),
        sa.Column("model_version", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # ── drift_events ──────────────────────────────────────────────────────────
    op.create_table(
        "drift_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("sport_id", sa.Integer, sa.ForeignKey("sports.id")),
        sa.Column("kind", sa.String(32)),
        sa.Column("metric_name", sa.String(64)),
        sa.Column("metric_value", sa.Numeric(10, 6)),
        sa.Column("threshold", sa.Numeric(10, 6)),
        sa.Column("action_taken", sa.String(128)),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # ── api_keys ──────────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("key_hash", sa.String(512), nullable=False),
        sa.Column("scopes", postgresql.ARRAY(sa.Text), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )

    # ── api_requests (partitioned by month) ───────────────────────────────────
    op.create_table(
        "api_requests",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("api_key_id", sa.Integer, sa.ForeignKey("api_keys.id", ondelete="SET NULL")),
        sa.Column("route", sa.String(256)),
        sa.Column("status", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("ip", sa.String(64)),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_api_requests_ts", "api_requests", ["ts"])
    op.create_index("ix_api_requests_key_id", "api_requests", ["api_key_id"])

    # ── pg_notify trigger on predictions ─────────────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION notify_new_prediction()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
          payload json;
        BEGIN
          payload := json_build_object(
            'game_id',          NEW.game_id,
            'prediction_id',    NEW.id,
            'target',           NEW.target,
            'probability',      NEW.probability,
            'is_lineup_update', false
          );
          PERFORM pg_notify('predictions_channel', payload::text);
          RETURN NEW;
        END;
        $$;
    """)
    op.execute("""
        CREATE TRIGGER predictions_notify
        AFTER INSERT ON predictions
        FOR EACH ROW EXECUTE FUNCTION notify_new_prediction();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS predictions_notify ON predictions;")
    op.execute("DROP FUNCTION IF EXISTS notify_new_prediction();")
    op.drop_table("api_requests")
    op.drop_table("api_keys")
    op.drop_table("drift_events")
    op.drop_table("predictions_audit")
    op.drop_table("predictions")
    op.drop_table("models")
    op.drop_table("matchup_features")
    op.drop_table("player_features")
    op.drop_table("team_features")
    op.drop_table("injuries")
    op.drop_table("lineups")
    op.drop_table("plays")
    op.drop_table("player_game_stats")
    op.drop_table("team_game_stats")
    op.drop_table("games")
    op.drop_table("venues")
    op.drop_table("players")
    op.drop_table("teams")
    op.drop_table("sports")
