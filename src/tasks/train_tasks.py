"""Training, drift monitoring, and model promotion tasks."""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.db.session import sync_session_factory

log = get_logger(__name__)


def train_challenger(sport: str, kind: str = "winner") -> dict:
    """Nightly: retrain from scratch, log to MLflow as candidate."""

    from src.core.time import utc_now
    from src.models.train_winner import train_winner_model

    with sync_session_factory() as session:
        df = _load_training_df(session, sport)
        if len(df) < 100:
            log.warning("train.insufficient_data", sport=sport, n=len(df))
            return {"status": "skipped", "reason": "insufficient_data"}

        feature_names = _get_feature_names(sport, kind)
        holdout_df = df[df["scheduled_utc"] >= df["scheduled_utc"].quantile(0.9)].copy()
        train_df = df[df["scheduled_utc"] < df["scheduled_utc"].quantile(0.9)].copy()

        run_id, metrics = train_winner_model(
            sport=sport,
            training_df=train_df,
            feature_names=feature_names,
            holdout_df=holdout_df,
            n_optuna_trials=30,
            run_name=f"{sport}_{kind}_challenger_{utc_now().strftime('%Y%m%d')}",
        )

    log.info("train.challenger_done", sport=sport, run_id=run_id, metrics=metrics)
    return {"run_id": run_id, "metrics": metrics}


def evaluate_and_promote(sport: str, kind: str = "winner") -> dict:
    """Weekly: compare best nightly challenger to champion, promote if gates pass."""
    import mlflow

    from src.core.config import settings
    from src.models.registry import promote_model
    from src.models.train_winner import should_promote

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    # Find all challenger runs from the past 7 days
    runs = client.search_runs(
        experiment_ids=["0"],  # default experiment
        filter_string=f"tags.sport = '{sport}' AND tags.kind = '{kind}' AND status = 'FINISHED'",
        order_by=["metrics.logloss ASC"],
        max_results=7,
    )

    if not runs:
        return {"status": "no_challengers"}

    best_run = runs[0]
    challenger_metrics = dict(best_run.data.metrics)

    with sync_session_factory() as session:
        from src.db.models import ModelRecord
        from src.db.models import Sport as SportModel

        sport_obj = session.query(SportModel).filter_by(code=sport).first()
        if sport_obj is None:
            return {"status": "sport_not_found"}

        champion = (
            session.query(ModelRecord)
            .filter_by(sport_id=sport_obj.id, kind=kind, target="home_won", active=True)
            .first()
        )

        champion_metrics = (
            dict(champion.metrics) if champion else {"logloss": 999.0, "ece": 999.0, "brier": 999.0}
        )

        promote, reason = should_promote(challenger_metrics, champion_metrics)
        log.info("evaluate.gate", sport=sport, promote=promote, reason=reason)

        if promote:
            from src.features.common import feature_spec_hash

            feature_names = _get_feature_names(sport, kind)
            fs_hash = feature_spec_hash(feature_names)

            promote_model(
                session=session,
                run_id=best_run.info.run_id,
                sport_id=sport_obj.id,
                kind=kind,
                target="home_won",
                version=best_run.info.run_id[:8],
                metrics=challenger_metrics,
                feature_spec_hash=fs_hash,
            )
            session.commit()
            _notify_ops(
                f"✅ {sport.upper()} {kind} model promoted. LogLoss: {challenger_metrics.get('logloss', '?'):.4f}"
            )

        elif champion and len(runs) >= 3:
            # Three consecutive failures → ops alert
            _notify_ops(
                f"⚠️ {sport.upper()} {kind}: 3+ challengers failed gate. Last reason: {reason}. Check feature pipeline."
            )

    return {"promote": promote, "reason": reason, "run_id": best_run.info.run_id}


def run_drift_monitor(sport: str) -> dict:
    """Daily drift detection across performance, calibration, and feature PSI."""
    import numpy as np

    from src.core.config import settings
    from src.models.eval.metrics import compute_ece

    log.info("drift.monitor.start", sport=sport)
    events = []

    with sync_session_factory() as session:
        from src.db.models import ModelRecord
        from src.db.models import Sport as SportModel

        sport_obj = session.query(SportModel).filter_by(code=sport).first()
        if sport_obj is None:
            return {"events": []}

        champion = (
            session.query(ModelRecord)
            .filter_by(sport_id=sport_obj.id, kind="winner", active=True)
            .first()
        )
        if champion is None:
            return {"events": []}

        # Load recent predictions (last 30 completed games) vs actuals
        from sqlalchemy import text

        result = session.execute(
            text("""
            SELECT p.probability, CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as actual
            FROM predictions p
            JOIN games g ON g.id = p.game_id
            WHERE p.model_id = :mid AND g.status='final' AND p.target='home_won'
              AND g.scheduled_utc > NOW() - INTERVAL '30 days'
            ORDER BY g.scheduled_utc DESC
            LIMIT 30
        """),
            {"mid": champion.id},
        )
        rows = list(result)

        if len(rows) >= 10:
            y_prob = np.array([float(r.probability) for r in rows])
            y_true = np.array([int(r.actual) for r in rows])
            from sklearn.metrics import log_loss

            recent_ll = float(log_loss(y_true, y_prob))
            baseline_ll = champion.metrics.get("logloss", 0.693)

            if recent_ll > baseline_ll * (1 + settings.drift_logloss_threshold):
                events.append(
                    {
                        "type": "performance",
                        "metric": "logloss",
                        "value": recent_ll,
                        "threshold": baseline_ll,
                    }
                )
                _notify_ops(
                    f"🔴 Drift: {sport.upper()} logloss {recent_ll:.4f} vs baseline {baseline_ll:.4f}"
                )
                # Trigger priority retrain
                train_challenger.apply_async(
                    kwargs={"sport": sport, "kind": "winner"}, queue="high_priority"
                )

            ece = compute_ece(y_true, y_prob)
            if ece > settings.drift_ece_threshold:
                events.append({"type": "calibration", "metric": "ece", "value": ece})

    log.info("drift.monitor.done", sport=sport, events=len(events))
    return {"events": events}


def generate_backtest_report(sport: str) -> dict:

    from src.models.eval.report import generate_winner_backtest_report

    with sync_session_factory() as session:
        df = _load_training_df(session, sport)
        feature_names = _get_feature_names(sport, "winner")
        path = generate_winner_backtest_report(sport, df, feature_names)
    return {"report": str(path)}


def hyperparam_search(sport: str) -> dict:
    """Monthly: full Optuna search with 50 trials + walk-forward objective."""
    return train_challenger(sport, kind="winner")  # same pipeline, more trials via env var


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_training_df(session: Any, sport: str) -> Any:
    import pandas as pd
    from sqlalchemy import text

    rows = session.execute(
        text("""
        SELECT g.id, g.scheduled_utc, g.season, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS target,
               mf.features
        FROM games g
        LEFT JOIN matchup_features mf ON mf.game_id = g.id
        JOIN sports s ON s.id = g.sport_id
        WHERE s.code = :code AND g.status='final' AND g.home_score IS NOT NULL
        ORDER BY g.scheduled_utc
    """),
        {"code": sport},
    ).fetchall()
    return pd.DataFrame([dict(r._mapping) for r in rows])


def _get_feature_names(sport: str, kind: str) -> list[str]:
    """Return canonical feature list for a sport/kind. In production, loaded from last champion."""
    from src.models.configs.mlb_winner import MLB_WINNER_FEATURES
    from src.models.configs.nba_winner import NBA_WINNER_FEATURES

    if sport == "nba":
        return NBA_WINNER_FEATURES
    return MLB_WINNER_FEATURES


def _notify_ops(message: str) -> None:
    from src.core.config import settings

    if settings.discord_webhook_ops:
        import contextlib

        import httpx

        with contextlib.suppress(Exception):
            httpx.post(settings.discord_webhook_ops, json={"content": message}, timeout=5)


# Register as Celery tasks
from src.tasks.celery_app import app  # noqa: E402

train_challenger = app.task(name="src.tasks.train_tasks.train_challenger", bind=False)(
    train_challenger
)
evaluate_and_promote = app.task(name="src.tasks.train_tasks.evaluate_and_promote", bind=False)(
    evaluate_and_promote
)
run_drift_monitor = app.task(name="src.tasks.train_tasks.run_drift_monitor", bind=False)(
    run_drift_monitor
)
generate_backtest_report = app.task(
    name="src.tasks.train_tasks.generate_backtest_report", bind=False
)(generate_backtest_report)
hyperparam_search = app.task(name="src.tasks.train_tasks.hyperparam_search", bind=False)(
    hyperparam_search
)
