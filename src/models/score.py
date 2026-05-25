"""Batch scoring: load active model, run inference on upcoming games, write predictions."""

from __future__ import annotations

import difflib
import json
from datetime import timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import as_of_for_game, utc_now
from src.db.models import Game, ModelRecord, Player, Prediction, PredictionAudit, Sport, Team
from src.ingest.common import dict_hash
from src.models.registry import load_model

log = get_logger(__name__)


def score_upcoming_games(session: Session, sport_code: str, hours_ahead: int = 48) -> int:
    """Score all upcoming games for a sport. Returns number of predictions written."""
    now = utc_now()
    window_end = now + timedelta(hours=hours_ahead)

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        log.error("score.sport_not_found", sport=sport_code)
        return 0

    _BETTABLE_STATUSES = [
        "scheduled",
        "pre-game",
        "in progress",
        "in_progress_inning_1",
        "delayed",
        "delayed start",
    ]
    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= now - timedelta(hours=4),
            Game.scheduled_utc <= window_end,
            Game.status.in_(_BETTABLE_STATUSES),
        )
        .all()
    )

    if not games:
        log.info("score.no_games", sport=sport_code)
        return 0

    # Load active winner model
    winner_model_record = (
        session.query(ModelRecord)
        .filter_by(sport_id=sport.id, kind="winner", target="home_won", active=True)
        .first()
    )

    count = 0
    for game in games:
        try:
            written = _score_game_winner(session, game, winner_model_record, sport)
            count += written
        except Exception as exc:
            log.error("score.game_failed", game_id=game.id, error=str(exc))

    session.flush()
    log.info("score.complete", sport=sport_code, predictions=count)
    return count


def _score_game_winner(
    session: Session,
    game: Game,
    model_record: ModelRecord | None,
    sport: Sport,
) -> int:
    if model_record is None:
        log.warning("score.no_active_model", sport=sport.code, kind="winner")
        return 0

    # Build features
    from src.features.mlb.matchup import build_matchup_features as mlb_matchup
    from src.features.nba.matchup import build_matchup_features as nba_matchup

    build_fn = nba_matchup if sport.code == "nba" else mlb_matchup
    as_of = as_of_for_game(game.scheduled_utc)
    try:
        features = build_fn(session=session, game=game, as_of=as_of)
    except Exception as exc:
        log.error("score.feature_build_failed", game_id=game.id, error=str(exc))
        return 0

    # Load model and infer
    try:
        model = load_model(model_record.mlflow_run_id, framework="sklearn")
    except Exception as exc:
        log.error("score.model_load_failed", run_id=model_record.mlflow_run_id, error=str(exc))
        return 0

    feature_names_raw = json.loads(load_model_feature_names(model_record.mlflow_run_id))
    X = np.array([[features.get(n, 0.0) for n in feature_names_raw]], dtype=np.float32)
    proba_home_win = float(model.predict_proba(X)[0, 1])

    # Per-feature SHAP-style leaf contributions — stored in audit for "got it right" posts.
    # XGBoost pred_contribs gives (n_samples, n_features+1); last col is bias.
    try:
        import xgboost as xgb

        dm = xgb.DMatrix(X, feature_names=feature_names_raw)
        raw_contribs = model.xgb_clf.get_booster().predict(dm, pred_contribs=True)
        contrib_dict = {
            feature_names_raw[i]: float(raw_contribs[0, i]) for i in range(len(feature_names_raw))
        }
        top_contribs = dict(
            sorted(contrib_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        )
        features["_contribs"] = top_contribs
    except Exception as exc:
        log.warning("score.contribs_failed", game_id=game.id, error=str(exc))

    features_hash = _hash_features({k: v for k, v in features.items() if not k.startswith("_")})

    # Upsert prediction
    _filter = dict(
        game_id=game.id,
        model_id=model_record.id,
        target="home_won",
        player_id=None,
    )
    existing = session.query(Prediction).filter_by(**_filter).first()

    now = utc_now()
    if existing is None:
        pred = Prediction(
            game_id=game.id,
            model_id=model_record.id,
            player_id=None,
            target="home_won",
            value=Decimal(str(round(proba_home_win, 4))),
            probability=Decimal(str(round(proba_home_win, 4))),
            features_hash=features_hash,
            created_at=now,
        )
        try:
            from sqlalchemy.exc import IntegrityError

            with session.begin_nested():
                session.add(pred)
                session.flush()
        except IntegrityError:
            # Race condition: another worker inserted between our SELECT and INSERT.
            # Re-fetch and fall through to the update path.
            existing = session.query(Prediction).filter_by(**_filter).first()
            if existing is None:
                return 0
            existing.probability = Decimal(str(round(proba_home_win, 4)))
            existing.value = Decimal(str(round(proba_home_win, 4)))
            existing.features_hash = features_hash
            existing.created_at = now
            return 1

        audit = PredictionAudit(
            prediction_id=pred.id,
            raw_features=features,
            model_version=model_record.version,
            created_at=now,
        )
        session.add(audit)

        # Notify via PG LISTEN/NOTIFY — payload contract must match listener.py
        session.execute(
            text("SELECT pg_notify('predictions_channel', :payload)"),
            {
                "payload": json.dumps(
                    {
                        "game_id": game.id,
                        "target": "home_won",
                        "probability": float(round(proba_home_win, 4)),
                        "is_lineup_update": False,
                    }
                )
            },
        )
        return 1
    else:
        existing.probability = Decimal(str(round(proba_home_win, 4)))
        existing.value = Decimal(str(round(proba_home_win, 4)))
        existing.features_hash = features_hash
        existing.created_at = now
        return 1


def load_model_feature_names(run_id: str) -> str:
    """Load feature_names.json from MLflow artifacts."""
    import mlflow

    from src.core.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    local_path = client.download_artifacts(run_id, "feature_names.json")
    with open(local_path) as f:
        data = json.load(f)
    return json.dumps(data.get("feature_names", []))


def _hash_features(features: dict[str, Any]) -> str:
    return dict_hash(features)


# ── Player props scoring ──────────────────────────────────────────────────────

_MLB_PITCHER_STATS = {"PITCHER_K", "PITCHER_ER"}


def score_props_upcoming(session: Session, sport_code: str, hours_ahead: int = 36) -> int:
    """Score player props for upcoming games. Returns number of predictions written."""
    from src.ingest.odds.draftkings import get_player_props

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        return 0

    try:
        dk_props = get_player_props(sport_code)
    except Exception as exc:
        log.error("score_props.dk_fetch_failed", sport=sport_code, error=str(exc))
        return 0

    if not dk_props:
        log.info("score_props.no_dk_lines", sport=sport_code)
        return 0

    # Pre-load all active props models, keyed by stat
    stats_needed = {p["stat"] for p in dk_props}
    prop_models: dict[str, tuple[ModelRecord, Any, list[str]]] = {}
    for stat in stats_needed:
        rec = (
            session.query(ModelRecord)
            .filter_by(sport_id=sport.id, kind="props", target=stat, active=True)
            .first()
        )
        if rec is None:
            continue
        try:
            m = load_model(rec.mlflow_run_id, framework="sklearn")
            fn = json.loads(load_model_feature_names(rec.mlflow_run_id))
            prop_models[stat] = (rec, m, fn)
        except Exception as exc:
            log.warning("score_props.model_load_failed", stat=stat, error=str(exc))

    if not prop_models:
        log.info("score_props.no_active_models", sport=sport_code)
        return 0

    now = utc_now()
    count = 0

    for prop in dk_props:
        stat = prop["stat"]
        if stat not in prop_models:
            continue
        rec, model, feat_names = prop_models[stat]

        player = _lookup_player_by_name(session, sport, prop["player_name"])
        if player is None:
            continue

        result = _find_player_game(session, player.id, sport, now, hours_ahead)
        if result is None:
            continue
        game, team, opp = result

        try:
            feats = _build_props_features(session, sport_code, player, team, opp, stat, now)
        except Exception as exc:
            log.warning(
                "score_props.feature_failed", player_id=player.id, stat=stat, error=str(exc)
            )
            continue

        if not feats:
            continue

        x = np.array([[feats.get(n, 0.0) for n in feat_names]], dtype=np.float32)
        try:
            quantile_preds = model.predict_row(x[0])
            median = float(model.predict(x)[0.50][0])
            dk_line = prop.get("line")
            p_over = model.implied_over_probability(x[0], dk_line) if dk_line else 0.5
        except Exception as exc:
            log.warning("score_props.infer_failed", player_id=player.id, stat=stat, error=str(exc))
            continue

        # Store quantiles + DK line metadata in quantiles dict
        quant_store: dict[str, Any] = {str(k): round(v, 2) for k, v in quantile_preds.items()}
        if dk_line is not None:
            quant_store["dk_line"] = float(dk_line)
        if prop.get("over_odds") is not None:
            quant_store["dk_over_odds"] = float(prop["over_odds"])
        if prop.get("under_odds") is not None:
            quant_store["dk_under_odds"] = float(prop["under_odds"])

        features_hash = _hash_features({k: v for k, v in feats.items()})

        _filter = dict(game_id=game.id, model_id=rec.id, target=stat, player_id=player.id)
        existing = session.query(Prediction).filter_by(**_filter).first()

        if existing is None:
            pred = Prediction(
                game_id=game.id,
                model_id=rec.id,
                player_id=player.id,
                target=stat,
                value=Decimal(str(round(median, 2))),
                probability=Decimal(str(round(p_over, 4))),
                quantiles=quant_store,
                features_hash=features_hash,
                created_at=now,
            )
            try:
                from sqlalchemy.exc import IntegrityError

                with session.begin_nested():
                    session.add(pred)
                    session.flush()
            except IntegrityError:
                existing = session.query(Prediction).filter_by(**_filter).first()
                if existing:
                    _update_prop_prediction(
                        existing, median, p_over, quant_store, features_hash, now
                    )
            else:
                count += 1
                # No pg_notify for individual props — they show up in `/props` command on demand
        else:
            _update_prop_prediction(existing, median, p_over, quant_store, features_hash, now)
            count += 1

    session.flush()
    log.info("score_props.complete", sport=sport_code, predictions=count)
    return count


def _update_prop_prediction(
    pred: Prediction,
    median: float,
    p_over: float,
    quant_store: dict[str, Any],
    features_hash: str,
    now: Any,
) -> None:
    pred.value = Decimal(str(round(median, 2)))
    pred.probability = Decimal(str(round(p_over, 4)))
    pred.quantiles = quant_store
    pred.features_hash = features_hash
    pred.created_at = now


def _lookup_player_by_name(session: Session, sport: Sport, name: str) -> Player | None:
    players = session.query(Player).filter_by(sport_id=sport.id).all()
    if not players:
        return None
    names = [p.full_name for p in players if p.full_name]
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if not matches:
        return None
    return next((p for p in players if p.full_name == matches[0]), None)


def _find_player_game(
    session: Session,
    player_id: int,
    sport: Sport,
    now: Any,
    hours_ahead: int,
) -> tuple[Game, Team, Team] | None:
    """Find player's next game. Returns (game, player_team, opponent_team) or None."""
    since = now - timedelta(hours=4)
    until = now + timedelta(hours=hours_ahead)
    _statuses = "('scheduled','pre-game','in progress','delayed','delayed start')"

    row = session.execute(
        text(f"""
            SELECT g.id AS game_id, g.home_team_id, g.away_team_id, pgs.team_id
            FROM player_game_stats pgs
            JOIN games g ON g.id = pgs.game_id
            WHERE pgs.player_id = :pid
              AND g.sport_id  = :sport_id
              AND g.status    IN {_statuses}
              AND g.scheduled_utc BETWEEN :since AND :until
            ORDER BY g.scheduled_utc LIMIT 1
        """),
        {"pid": player_id, "sport_id": sport.id, "since": since, "until": until},
    ).fetchone()

    if row is None:
        # Fallback: find via most recent team
        team_row = session.execute(
            text("""
                SELECT pgs.team_id FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE pgs.player_id = :pid AND g.sport_id = :sport_id
                ORDER BY g.scheduled_utc DESC LIMIT 1
            """),
            {"pid": player_id, "sport_id": sport.id},
        ).fetchone()
        if team_row is None:
            return None
        team_id = team_row.team_id
        game_row = session.execute(
            text(f"""
                SELECT id, home_team_id, away_team_id FROM games
                WHERE sport_id = :sport_id
                  AND (home_team_id = :tid OR away_team_id = :tid)
                  AND status IN {_statuses}
                  AND scheduled_utc BETWEEN :since AND :until
                ORDER BY scheduled_utc LIMIT 1
            """),
            {"sport_id": sport.id, "tid": team_id, "since": since, "until": until},
        ).fetchone()
        if game_row is None:
            return None
        game = session.get(Game, game_row.id)
        team = session.get(Team, team_id)
        opp_id = (
            game_row.away_team_id if team_id == game_row.home_team_id else game_row.home_team_id
        )
        opp = session.get(Team, opp_id)
    else:
        game = session.get(Game, row.game_id)
        team = session.get(Team, row.team_id)
        opp_id = row.away_team_id if row.team_id == row.home_team_id else row.home_team_id
        opp = session.get(Team, opp_id)

    if game is None or team is None or opp is None:
        return None
    return game, team, opp


def _build_props_features(
    session: Session,
    sport_code: str,
    player: Player,
    team: Team,
    opp: Team,
    stat: str,
    now: Any,
) -> dict[str, Any]:
    if sport_code == "nba":
        from src.features.nba.player import build_player_features

        return build_player_features(
            session=session,
            player_id=player.id,
            team_id=team.id,
            opponent_team_id=opp.id,
            as_of_utc=now,
            stat=stat,
        )
    elif sport_code == "mlb":
        if stat in _MLB_PITCHER_STATS:
            from src.features.mlb.player import build_pitcher_features

            return build_pitcher_features(
                session=session,
                player_id=player.id,
                as_of_utc=now,
                stat=stat,  # pass full name e.g. PITCHER_K so features match training
            )
        else:
            from src.features.mlb.player import build_batter_features

            return build_batter_features(
                session=session,
                player_id=player.id,
                opponent_pitcher_throws="R",
                stadium_factor=1.0,
                as_of_utc=now,
                stat=stat,
            )
    return {}
