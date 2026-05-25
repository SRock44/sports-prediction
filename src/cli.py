"""Typer CLI: backfill, train, eval, keys, model management.

Usage:
  python -m src.cli backfill --sport nba --seasons 5
  python -m src.cli train --sport nba --kind winner
  python -m src.cli eval --sport nba --kind winner
  python -m src.cli keys create --name "discord-bot"
  python -m src.cli keys list
  python -m src.cli keys revoke <key-id>
  python -m src.cli model list --sport nba
  python -m src.cli model rollback --sport nba --kind winner
  python -m src.cli score --sport nba
"""

from __future__ import annotations

from datetime import UTC, date, timedelta
from typing import Annotated, Any

import typer

app = typer.Typer(name="prediction", add_completion=False, help="Sports prediction CLI")
keys_app = typer.Typer(help="API key management")
model_app = typer.Typer(help="Model management")
app.add_typer(keys_app, name="keys")
app.add_typer(model_app, name="model")

# ── Backfill ──────────────────────────────────────────────────────────────────


@app.command()
def backfill(
    sport: Annotated[str, typer.Option("--sport", help="nba or mlb")] = "nba",
    seasons: Annotated[int, typer.Option("--seasons", help="Number of past seasons")] = 5,
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run")] = False,
) -> None:
    """Backfill historical game data + box scores for N seasons."""
    from src.core.time import mlb_season_for_date, nba_season_for_date
    from src.db.session import get_sync_session

    today = date.today()
    if sport == "nba":
        current_season = nba_season_for_date(today)
        season_years = list(range(current_season - seasons + 1, current_season + 1))
    elif sport == "mlb":
        current_season = mlb_season_for_date(today)
        season_years = list(range(current_season - seasons + 1, current_season + 1))
    else:
        typer.echo(f"Unknown sport: {sport}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Backfilling {sport.upper()} seasons: {season_years}")
    if dry_run:
        typer.echo("[dry-run] Would ingest schedule + box scores for listed seasons.")
        raise typer.Exit(0)

    with get_sync_session() as session:
        total_inserted = total_updated = 0

        if sport == "nba":
            from src.ingest.nba.games import ingest_season_schedule
            from src.ingest.nba.players import sync_players, sync_teams

            r = sync_teams(session)
            typer.echo(f"  Teams: +{r.rows_inserted} updated={r.rows_updated}")
            session.commit()

            for season_year in season_years:
                typer.echo(f"  Season {season_year}…", nl=False)
                r = ingest_season_schedule(session, season_year)
                session.commit()
                typer.echo(f" schedule +{r.rows_inserted} upd={r.rows_updated}", nl=False)

                r2 = sync_players(session, season_year)
                session.commit()
                typer.echo(f"  players +{r2.rows_inserted}", nl=False)

                total_inserted += r.rows_inserted + r2.rows_inserted
                total_updated += r.rows_updated + r2.rows_updated
                typer.echo()

        elif sport == "mlb":
            from src.ingest.mlb.games import ingest_season_schedule  # type: ignore[assignment]

            for season_year in season_years:
                typer.echo(f"  Season {season_year}…", nl=False)
                r = ingest_season_schedule(session, season_year)
                session.commit()
                typer.echo(f" schedule +{r.rows_inserted} upd={r.rows_updated}")

                total_inserted += r.rows_inserted
                total_updated += r.rows_updated

    typer.echo(f"\nDone. Total inserted={total_inserted} updated={total_updated}")


# ── Train ─────────────────────────────────────────────────────────────────────


@app.command()
def train(
    sport: Annotated[str, typer.Option("--sport")] = "nba",
    kind: Annotated[str, typer.Option("--kind", help="winner or props")] = "winner",
    promote: Annotated[bool, typer.Option("--promote/--no-promote")] = False,
    trials: Annotated[int, typer.Option("--trials")] = 200,
) -> None:
    """Train a challenger model. Optionally promote if it beats champion."""
    from src.db.session import get_sync_session

    typer.echo(f"Training {sport.upper()} {kind} model ({trials} Optuna trials)…")

    with get_sync_session() as session:
        training_df, holdout_df, feature_names = _load_training_data(session, sport, kind)

    if training_df.empty:
        typer.echo("No training data found. Run backfill first.", err=True)
        raise typer.Exit(1)

    typer.echo(f"  Training rows: {len(training_df)}  Holdout rows: {len(holdout_df)}")

    if kind == "winner":
        from src.models.train_winner import should_promote, train_winner_model

        run_id, metrics = train_winner_model(sport, training_df, feature_names, holdout_df, trials)
        typer.echo(f"  Run ID: {run_id}")
        typer.echo(f"  Metrics: {metrics}")

        if promote:
            from src.db.models import ModelRecord
            from src.db.models import Sport as SportModel
            from src.features.common import feature_spec_hash as compute_fsh

            with get_sync_session() as session:
                sport_obj = session.query(SportModel).filter_by(code=sport).first()
                champion = (
                    (
                        session.query(ModelRecord)
                        .filter_by(
                            sport_id=sport_obj.id if sport_obj else 0, kind="winner", active=True
                        )
                        .first()
                    )
                    if sport_obj
                    else None
                )
                champion_metrics = champion.metrics if champion else {}
                sport_id = sport_obj.id if sport_obj else None

            ok, reason = should_promote(metrics, champion_metrics)
            if ok and sport_id is not None:
                from src.models.registry import promote_model

                with get_sync_session() as session:
                    promote_model(
                        session,
                        run_id,
                        sport_id,
                        kind,
                        "home_won",
                        version=run_id[:12],
                        metrics=metrics,
                        feature_spec_hash=compute_fsh(feature_names),
                    )
                    session.commit()
                typer.echo(f"  Promoted. {reason}")
            else:
                typer.echo(f"  Not promoted: {reason}")

    elif kind == "props":
        from src.db.models import Sport as SportModel
        from src.features.common import feature_spec_hash as compute_fsh
        from src.models.registry import promote_model
        from src.models.train_props import train_props_model

        with get_sync_session() as session:
            sport_obj = session.query(SportModel).filter_by(code=sport).first()
            sport_id = sport_obj.id if sport_obj else None

        if sport_id is None:
            typer.echo(f"Sport '{sport}' not found in DB.", err=True)
            raise typer.Exit(1)

        for stat in _props_stats(sport):
            typer.echo(f"  Prop stat: {stat}…", nl=False)
            try:
                with get_sync_session() as prop_session:
                    prop_df, hold_df, prop_feature_names = _load_props_training_data(
                        prop_session, sport, stat
                    )
                if prop_df.empty:
                    typer.echo(" no data, skip")
                    continue
                typer.echo(f" {len(prop_df)} train / {len(hold_df)} holdout rows…", nl=False)
                run_id, metrics = train_props_model(
                    sport, stat, prop_df, prop_feature_names, hold_df
                )
                with get_sync_session() as session:
                    promote_model(
                        session,
                        run_id,
                        sport_id,
                        kind="props",
                        target=stat,
                        version=run_id[:12],
                        metrics=metrics,
                        feature_spec_hash=compute_fsh(prop_feature_names),
                    )
                    session.commit()
                typer.echo(f" done run={run_id[:8]} mae={metrics.get('mae_median', '?'):.3f}")
            except Exception as exc:
                typer.echo(f" ERROR: {exc}", err=True)


def _props_stats(sport: str) -> list[str]:
    if sport == "nba":
        return ["PTS", "REB", "AST", "3PM", "PRA"]
    elif sport == "mlb":
        return ["H", "HR", "TB", "RBI", "PITCHER_K", "PITCHER_ER"]
    return []


def _load_training_data(
    session: Any,
    sport: str,
    kind: str,
) -> tuple[Any, Any, list[str]]:
    """Load training + holdout DataFrames for a given sport/kind."""
    import pandas as pd

    from src.db.models import Game, MatchupFeature, Sport

    sport_obj = session.query(Sport).filter_by(code=sport).first()
    if sport_obj is None:
        return pd.DataFrame(), pd.DataFrame(), []

    from sqlalchemy import or_

    rows = (
        session.query(Game, MatchupFeature)
        .join(MatchupFeature, MatchupFeature.game_id == Game.id)
        .filter(
            Game.sport_id == sport_obj.id,
            Game.status.in_(["final"]),
            # Regular season only: NBA stores no game_type (all regular + playoffs
            # mixed as NULL); MLB uses 'R' for regular season.
            or_(
                Game.meta["game_type"] == None,  # noqa: E711 — SQLAlchemy IS NULL
                Game.meta["game_type"].astext == "R",
            ),
        )
        .order_by(Game.scheduled_utc)
        .all()
    )

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), []

    records = []
    for game, mf in rows:
        if not mf.features:
            continue
        rec = dict(mf.features)
        rec["y"] = 1 if (game.home_score or 0) > (game.away_score or 0) else 0
        rec["game_date"] = game.scheduled_utc.date() if game.scheduled_utc else None
        rec["season"] = game.season
        records.append(rec)

    df = pd.DataFrame(records).dropna(subset=["game_date"])
    if df.empty:
        return df, df, []

    # Use the most recent season as holdout for a proper out-of-sample evaluation.
    # Fall back to 8-week window only if there is just one season of data.
    max_season = df["season"].max()
    training_df = df[df["season"] < max_season]
    holdout_df = df[df["season"] == max_season]
    if training_df.empty:
        cutoff = df["game_date"].max() - timedelta(weeks=8)
        training_df = df[df["game_date"] <= cutoff]
        holdout_df = df[df["game_date"] > cutoff]

    feature_names = [c for c in df.columns if c not in ("y", "game_date", "season")]
    return training_df, holdout_df, feature_names


def _nba_stat_sql(stat: str) -> str:
    """SQL expression to extract NBA stat value from player_game_stats.stats JSONB."""
    mapping = {
        "PTS": "stats->'traditional'->>'points'",
        "REB": "stats->'traditional'->>'reboundsTotal'",
        "AST": "stats->'traditional'->>'assists'",
        "3PM": "stats->'traditional'->>'threePointersMade'",
        "PRA": (
            "COALESCE((stats->'traditional'->>'points')::float, 0) + "
            "COALESCE((stats->'traditional'->>'reboundsTotal')::float, 0) + "
            "COALESCE((stats->'traditional'->>'assists')::float, 0)"
        ),
    }
    return mapping.get(stat, "NULL")


def _mlb_stat_sql(stat: str) -> tuple[str, str]:
    """SQL expressions (target_sql, minutes_sql) for MLB player stats."""
    batter = {
        "H": "stats->'batting'->>'hits'",
        "HR": "stats->'batting'->>'homeRuns'",
        "TB": "stats->'batting'->>'totalBases'",
        "RBI": "stats->'batting'->>'rbi'",
        "K": "stats->'batting'->>'strikeOuts'",
    }
    pitcher = {
        "PITCHER_K": "stats->'pitching'->>'strikeOuts'",
        "PITCHER_ER": "stats->'pitching'->>'earnedRuns'",
    }
    if stat in batter:
        return batter[stat], "'9'"
    if stat in pitcher:
        return pitcher[stat], "stats->'pitching'->>'inningsPitched'"
    return "NULL", "'0'"


def _parse_minutes_str(m: str | None) -> float:
    """Parse 'MM:SS' or decimal string to float minutes."""
    try:
        if m and ":" in str(m):
            parts = str(m).split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(m) if m else 0.0
    except (ValueError, TypeError, IndexError):
        return 0.0


def _load_props_training_data(
    session: Any,
    sport: str,
    stat: str,
) -> tuple[Any, Any, list[str]]:
    """Load training + holdout DataFrames for a player prop stat.

    Pulls raw player-game rows from player_game_stats, computes rolling features
    in pandas (matching build_player_features output), and returns:
      training_df, holdout_df, feature_names
    One row per (player, game); target = actual stat value in that game.
    """
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from src.db.models import Sport as SportModel

    sport_obj = session.query(SportModel).filter_by(code=sport).first()
    if sport_obj is None:
        return pd.DataFrame(), pd.DataFrame(), []

    if sport == "nba":
        target_expr = _nba_stat_sql(stat)
        min_expr = "stats->'traditional'->>'minutes'"
    else:
        target_expr, min_expr = _mlb_stat_sql(stat)

    if target_expr == "NULL":
        return pd.DataFrame(), pd.DataFrame(), []

    q = text(f"""
        SELECT
            pgs.player_id,
            g.id           AS game_id,
            g.scheduled_utc,
            g.season,
            CASE WHEN pgs.team_id = g.home_team_id THEN 1.0 ELSE 0.0 END AS is_home,
            ({target_expr})::float    AS target_val,
            ({min_expr})::text        AS minutes_raw
        FROM player_game_stats pgs
        JOIN games  g  ON g.id  = pgs.game_id
        JOIN sports sp ON sp.id = g.sport_id
        WHERE sp.code = :sport
          AND g.status = 'final'
          AND ({target_expr}) IS NOT NULL
        ORDER BY pgs.player_id, g.scheduled_utc
    """)
    rows = session.execute(q, {"sport": sport}).mappings().all()
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), []

    df = pd.DataFrame(rows)
    df["minutes"] = df["minutes_raw"].apply(_parse_minutes_str)
    # Filter out DNP rows (< 5 min played) — not useful training signal
    df = df[df["minutes"] >= 5].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    df["per_min"] = np.where(df["minutes"] > 0, df["target_val"] / df["minutes"], 0.0)
    df = df.sort_values(["player_id", "scheduled_utc"]).reset_index(drop=True)

    # Rolling features — shift(1) ensures we only use prior games (no leakage)
    grp = df.groupby("player_id", sort=False)

    def _roll_mean(s: Any, w: int) -> Any:
        return s.shift(1).rolling(w, min_periods=1).mean()

    def _roll_std(s: Any, w: int) -> Any:
        return s.shift(1).rolling(w, min_periods=3).std().fillna(0.0)

    for w in [5, 10, 20]:
        df[f"{stat}_last{w}"] = grp["target_val"].transform(lambda s, w=w: _roll_mean(s, w))
        df[f"{stat}_per_min_last{w}"] = grp["per_min"].transform(lambda s, w=w: _roll_mean(s, w))

    df[f"{stat}_std_last10"] = grp["target_val"].transform(lambda s: _roll_std(s, 10))
    df["minutes_last5"] = grp["minutes"].transform(lambda s: _roll_mean(s, 5)).fillna(20.0)
    df["minutes_last10"] = grp["minutes"].transform(lambda s: _roll_mean(s, 10)).fillna(20.0)

    # Rest days (days since last game, capped at 10)
    df["prev_utc"] = grp["scheduled_utc"].transform(lambda s: s.shift(1))
    df["rest_days"] = (
        (pd.to_datetime(df["scheduled_utc"]) - pd.to_datetime(df["prev_utc"]))
        .dt.total_seconds()
        .div(86400)
        .clip(0, 10)
        .fillna(2.0)
    )

    # Game number within player's history (for cold-start weight)
    df["_game_num"] = grp.cumcount()
    df["season_game_weight"] = (df["_game_num"] / 20.0).clip(0, 1.0)

    # Static placeholders matching build_player_features output
    df["home_away_split"] = df["is_home"]
    df["opp_def_rtg_at_pos"] = 0.0
    df["injury_status"] = 1.0
    df["target"] = df["target_val"]

    # Require at least 10 prior games for meaningful rolling features
    df = df[df["_game_num"] >= 10].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    feature_cols = [
        f"{stat}_last5",
        f"{stat}_last10",
        f"{stat}_last20",
        f"{stat}_per_min_last5",
        f"{stat}_per_min_last10",
        f"{stat}_per_min_last20",
        f"{stat}_std_last10",
        "minutes_last5",
        "minutes_last10",
        "home_away_split",
        "opp_def_rtg_at_pos",
        "rest_days",
        "injury_status",
        "season_game_weight",
    ]

    # Holdout = most recent season; training = everything before it
    max_season = df["season"].max()
    train_df = df[df["season"] < max_season].copy()
    hold_df = df[df["season"] == max_season].copy()
    if train_df.empty:
        cutoff = pd.to_datetime(df["scheduled_utc"]).max() - pd.Timedelta(weeks=8)
        train_df = df[pd.to_datetime(df["scheduled_utc"]) <= cutoff].copy()
        hold_df = df[pd.to_datetime(df["scheduled_utc"]) > cutoff].copy()

    return train_df, hold_df, feature_cols


# ── Eval ──────────────────────────────────────────────────────────────────────


@app.command()
def eval(
    sport: Annotated[str, typer.Option("--sport")] = "nba",
    kind: Annotated[str, typer.Option("--kind")] = "winner",
) -> None:
    """Run walk-forward backtest and write report to reports/."""
    typer.echo(f"Running walk-forward eval for {sport.upper()} {kind}…")
    from src.db.session import get_sync_session
    from src.models.eval.report import generate_winner_backtest_report

    with get_sync_session() as session:
        training_df, _, feature_names = _load_training_data(session, sport, kind)

    if training_df.empty:
        typer.echo("No data. Run backfill first.", err=True)
        raise typer.Exit(1)

    report_path = generate_winner_backtest_report(sport, training_df, feature_names)
    typer.echo(f"Report written to: {report_path}")


# ── Score ─────────────────────────────────────────────────────────────────────


@app.command()
def score(
    sport: Annotated[str, typer.Option("--sport")] = "nba",
    hours: Annotated[int, typer.Option("--hours")] = 48,
) -> None:
    """Score upcoming games for a sport."""
    from src.db.session import get_sync_session
    from src.models.score import score_upcoming_games

    typer.echo(f"Scoring upcoming {sport.upper()} games (next {hours}h)…")
    with get_sync_session() as session:
        n = score_upcoming_games(session, sport, hours_ahead=hours)
        session.commit()
    typer.echo(f"Scored {n} games.")


# ── Keys ──────────────────────────────────────────────────────────────────────


@keys_app.command("create")
def keys_create(
    name: Annotated[str, typer.Option("--name", "-n")] = "default",
    scopes: Annotated[str, typer.Option("--scopes")] = "predictions:read",
    expires_days: Annotated[int | None, typer.Option("--expires-days")] = None,
) -> None:
    """Generate a new API key and print the plaintext once."""
    from datetime import datetime

    from src.core.security import generate_api_key, hash_api_key
    from src.db.models.auth import ApiKey
    from src.db.session import get_sync_session

    plaintext = generate_api_key()
    hashed = hash_api_key(plaintext)
    scope_list = [s.strip() for s in scopes.split(",") if s.strip()]

    expires_at = None
    if expires_days is not None:
        expires_at = datetime.now(UTC) + timedelta(days=expires_days)

    with get_sync_session() as session:
        key = ApiKey(
            name=name,
            key_prefix=plaintext[:8],
            key_hash=hashed,
            scopes=scope_list,
            created_at=datetime.now(UTC),
            expires_at=expires_at,
        )
        session.add(key)
        session.commit()
        session.refresh(key)
        key_id = key.id

    typer.echo(f"\nAPI Key created (id={key_id}, name={name})")
    typer.echo(f"Scopes: {scope_list}")
    typer.echo(f"\n  KEY (save this — shown once):\n\n  {plaintext}\n")


@keys_app.command("list")
def keys_list() -> None:
    """List all active API keys (hashes not shown)."""
    from src.db.models.auth import ApiKey
    from src.db.session import get_sync_session

    with get_sync_session() as session:
        keys = session.query(ApiKey).order_by(ApiKey.created_at).all()

    if not keys:
        typer.echo("No keys found.")
        return

    typer.echo(f"\n{'ID':<5} {'Name':<20} {'Scopes':<30} {'Active':<8} {'Expires'}")
    typer.echo("-" * 75)
    for k in keys:
        active = "yes" if k.is_active else "no"
        expires = k.expires_at.date().isoformat() if k.expires_at else "never"
        typer.echo(f"{k.id:<5} {k.name:<20} {','.join(k.scopes):<30} {active:<8} {expires}")


@keys_app.command("revoke")
def keys_revoke(
    key_id: Annotated[int, typer.Argument(help="ID from 'keys list'")],
) -> None:
    """Revoke an API key immediately."""
    from datetime import datetime

    from src.db.models.auth import ApiKey
    from src.db.session import get_sync_session

    with get_sync_session() as session:
        k = session.get(ApiKey, key_id)
        if k is None:
            typer.echo(f"Key {key_id} not found.", err=True)
            raise typer.Exit(1)
        k.revoked_at = datetime.now(UTC)
        session.commit()

    typer.echo(f"Key {key_id} ({k.name}) revoked.")


# ── Model management ──────────────────────────────────────────────────────────


@model_app.command("list")
def model_list(
    sport: Annotated[str, typer.Option("--sport")] = "nba",
) -> None:
    """List model versions for a sport."""
    from src.db.models import ModelRecord, Sport
    from src.db.session import get_sync_session

    with get_sync_session() as session:
        sport_obj = session.query(Sport).filter_by(code=sport).first()
        if sport_obj is None:
            typer.echo(f"Sport '{sport}' not found.")
            raise typer.Exit(1)

        records = (
            session.query(ModelRecord)
            .filter_by(sport_id=sport_obj.id)
            .order_by(ModelRecord.trained_at.desc())
            .limit(20)
            .all()
        )

    if not records:
        typer.echo("No models found. Run train first.")
        return

    typer.echo(f"\n{'ID':<5} {'Kind':<10} {'Target':<12} {'Version':<10} {'Active':<8} {'Trained'}")
    typer.echo("-" * 70)
    for m in records:
        active = "* ACTIVE" if m.active else ""
        trained = m.trained_at.date().isoformat() if m.trained_at else "?"
        typer.echo(f"{m.id:<5} {m.kind:<10} {m.target:<12} {m.version:<10} {active:<8} {trained}")


@model_app.command("rollback")
def model_rollback(
    sport: Annotated[str, typer.Option("--sport")] = "nba",
    kind: Annotated[str, typer.Option("--kind")] = "winner",
) -> None:
    """Roll back to the previous active model version."""
    from src.db.models import Sport as SportModel
    from src.db.session import get_sync_session
    from src.models.registry import rollback_model

    with get_sync_session() as session:
        sport_obj = session.query(SportModel).filter_by(code=sport).first()
        if sport_obj is None:
            typer.echo(f"Sport '{sport}' not found.", err=True)
            raise typer.Exit(1)
        success = rollback_model(session, sport_obj.id, kind, "home_won")
        session.commit()

    if success:
        typer.echo(f"Rolled back {sport.upper()} {kind} model.")
    else:
        typer.echo("No previous model to roll back to.", err=True)


@model_app.command("promote")
def model_promote(
    run_id: Annotated[str, typer.Argument(help="MLflow run_id to promote")],
    sport: Annotated[str, typer.Option("--sport")] = "nba",
    kind: Annotated[str, typer.Option("--kind")] = "winner",
    target: Annotated[str, typer.Option("--target")] = "home_win",
) -> None:
    """Force-promote an MLflow run as the active model (skips promotion gate)."""
    import json

    from src.db.models import Sport as SportModel
    from src.db.session import get_sync_session
    from src.features.common import feature_spec_hash as compute_fsh
    from src.models.registry import get_run_metrics, promote_model
    from src.models.score import load_model_feature_names

    with get_sync_session() as session:
        sport_obj = session.query(SportModel).filter_by(code=sport).first()
        if sport_obj is None:
            typer.echo(f"Sport '{sport}' not found.", err=True)
            raise typer.Exit(1)

        run_metrics = get_run_metrics(run_id)
        try:
            feat_names = json.loads(load_model_feature_names(run_id))
        except Exception:
            feat_names = []

        promote_model(
            session,
            run_id,
            sport_obj.id,
            kind,
            target,
            version=run_id[:12],
            metrics=run_metrics,
            feature_spec_hash=compute_fsh(feat_names),
        )
        session.commit()

    typer.echo(f"Promoted run {run_id[:12]}… as active {sport} {kind} model.")


if __name__ == "__main__":
    app()
