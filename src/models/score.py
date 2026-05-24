"""Batch scoring: load active model, run inference on upcoming games, write predictions."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import as_of_for_game, utc_now
from src.db.models import Game, ModelRecord, Prediction, PredictionAudit, Sport
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
    existing = (
        session.query(Prediction)
        .filter_by(
            game_id=game.id,
            model_id=model_record.id,
            target="home_won",
            player_id=None,
        )
        .first()
    )

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
        session.add(pred)
        session.flush()

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
