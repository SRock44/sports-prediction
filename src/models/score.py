"""Batch scoring: load active model, run inference on upcoming games, write predictions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import utc_now, as_of_for_game
from src.db.models import Game, Sport, ModelRecord, Prediction, PredictionAudit
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

    games = session.query(Game).filter(
        Game.sport_id == sport.id,
        Game.scheduled_utc >= now,
        Game.scheduled_utc <= window_end,
        Game.status.in_(["scheduled", "pre-game"]),
    ).all()

    if not games:
        log.info("score.no_games", sport=sport_code)
        return 0

    # Load active winner model
    winner_model_record = session.query(ModelRecord).filter_by(
        sport_id=sport.id, kind="winner", target="home_won", active=True
    ).first()

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
    from src.features.nba.matchup import build_matchup_features as nba_matchup
    from src.features.mlb.matchup import build_matchup_features as mlb_matchup

    build_fn = nba_matchup if sport.code == "nba" else mlb_matchup
    try:
        features = build_fn(
            session=session,
            game_id=game.id,
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            scheduled_utc=game.scheduled_utc,
            sport_id=sport.id,
        )
    except Exception as exc:
        log.error("score.feature_build_failed", game_id=game.id, error=str(exc))
        return 0

    # Load model and infer
    try:
        model = load_model(model_record.mlflow_run_id, framework="sklearn")
    except Exception as exc:
        log.error("score.model_load_failed", run_id=model_record.mlflow_run_id, error=str(exc))
        return 0

    feature_names_raw = json.loads(
        load_model_feature_names(model_record.mlflow_run_id)
    )
    X = np.array([[features.get(n, 0.0) for n in feature_names_raw]], dtype=np.float32)
    proba_home_win = float(model.predict_proba(X)[0, 1])

    features_hash = _hash_features(features)

    # Upsert prediction
    existing = session.query(Prediction).filter_by(
        game_id=game.id,
        model_id=model_record.id,
        target="home_won",
        player_id=None,
    ).first()

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

        # Notify via PG LISTEN/NOTIFY
        session.execute(
            text("SELECT pg_notify('predictions_channel', :payload)"),
            {"payload": json.dumps({"prediction_id": pred.id, "game_id": game.id})},
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
    serialised = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]
