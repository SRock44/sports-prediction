"""Training, drift monitoring, and model promotion tasks."""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.db.session import sync_session_factory

log = get_logger(__name__)


def train_challenger(sport: str, kind: str = "winner", wide_search: bool = False) -> dict:
    """Nightly: retrain on all completed games, use most-recent season as holdout.

    wide_search=False (default) uses constrained HP bounds — prevents the Run-4
    overfitting where Optuna found max_depth=10/n_estimators=4115 that looked
    great on CV but regressed 0.022 log-loss on holdout.
    wide_search=True is used by the monthly hyperparam_search only.
    """
    from src.core.time import utc_now
    from src.models.train_winner import train_winner_model

    with sync_session_factory() as session:
        df = _load_training_df(session, sport)
        if len(df) < 100:
            log.warning("train.insufficient_data", sport=sport, n=len(df))
            return {"status": "skipped", "reason": "insufficient_data"}

        # Use champion's feature list so challenger is always evaluated on same features
        feature_names = _get_feature_names(sport, kind)

        # Season-based split — same as CLI and champion evaluation.
        # This ensures nightly challenger is compared on an identical holdout basis.
        max_season = df["season"].max()
        train_df = df[df["season"] < max_season].copy()
        holdout_df = df[df["season"] == max_season].copy()

        if train_df.empty or holdout_df.empty:
            # Single-season fallback: last 15% by time
            cutoff = int(len(df) * 0.85)
            train_df = df.iloc[:cutoff].copy()
            holdout_df = df.iloc[cutoff:].copy()

        run_id, metrics = train_winner_model(
            sport=sport,
            training_df=train_df,
            feature_names=feature_names,
            holdout_df=holdout_df,
            n_optuna_trials=100,
            run_name=f"{sport}_{kind}_challenger_{utc_now().strftime('%Y%m%d')}",
            wide_search=wide_search,
        )

    log.info("train.challenger_done", sport=sport, run_id=run_id, metrics=metrics)
    _append_training_log(sport, "Challenger Train", run_id, metrics)
    _notify_ops(
        f"🤖 {sport.upper()} challenger trained. "
        f"LogLoss: {metrics.get('logloss', '?'):.4f}  Acc: {metrics.get('accuracy', 0):.1%}"
    )
    return {"run_id": run_id, "metrics": metrics}


def evaluate_and_promote(sport: str, kind: str = "winner") -> dict:
    """Weekly: compare best nightly challenger to champion, promote if gates pass."""
    import mlflow

    from src.core.config import settings
    from src.models.registry import promote_model
    from src.models.train_winner import should_promote

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    # Resolve experiment ID by name — runs are logged under settings.mlflow_experiment_name
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        return {"status": "no_challengers"}
    exp_ids = [experiment.experiment_id]

    runs = client.search_runs(
        experiment_ids=exp_ids,
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
            _append_training_log(
                sport,
                "Challenger Promoted ✅",
                best_run.info.run_id,
                challenger_metrics,
                notes=f"Beat champion. Champion was {champion_metrics.get('logloss', '?'):.4f}, challenger {challenger_metrics.get('logloss', '?'):.4f}",
            )
            _notify_ops(
                f"✅ {sport.upper()} {kind} model promoted. "
                f"LogLoss: {champion_metrics.get('logloss', '?'):.4f} → {challenger_metrics.get('logloss', '?'):.4f}"
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
    """Monthly: full Optuna search with wide bounds — the only place wide_search=True is used."""
    return train_challenger(sport, kind="winner", wide_search=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_training_df(session: Any, sport: str) -> Any:
    import pandas as pd
    from sqlalchemy import text

    rows = session.execute(
        text("""
        SELECT g.id, g.scheduled_utc, g.season,
               CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS y,
               mf.features
        FROM games g
        JOIN matchup_features mf ON mf.game_id = g.id
        JOIN sports s ON s.id = g.sport_id
        WHERE s.code = :code AND g.status='final' AND g.home_score IS NOT NULL
          AND mf.features IS NOT NULL
        ORDER BY g.scheduled_utc
    """),
        {"code": sport},
    ).fetchall()

    records = []
    for r in rows:
        row = dict(r._mapping)
        features = row.pop("features") or {}
        rec = dict(features)
        rec["y"] = row["y"]
        rec["season"] = row["season"]
        rec["game_date"] = row["scheduled_utc"].date() if row["scheduled_utc"] else None
        records.append(rec)

    return pd.DataFrame(records).dropna(subset=["game_date"])


def _get_feature_names(sport: str, kind: str) -> list[str]:
    """Return canonical feature list for a sport/kind. In production, loaded from last champion."""
    from src.models.configs.mlb_winner import MLB_WINNER_FEATURES
    from src.models.configs.nba_winner import NBA_WINNER_FEATURES

    if sport == "nba":
        return NBA_WINNER_FEATURES
    return MLB_WINNER_FEATURES


def _append_training_log(
    sport: str,
    run_type: str,
    run_id: str,
    metrics: dict,
    params: dict | None = None,
    notes: str = "",
) -> None:
    """Append a structured entry to /app/reports/training_log.md (persists in Docker volume)."""
    import os
    from datetime import UTC, datetime

    os.makedirs("/app/reports", exist_ok=True)
    log_path = "/app/reports/training_log.md"

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ll = metrics.get("logloss", "?")
    acc = metrics.get("accuracy", 0)
    brier = metrics.get("brier", "?")
    ece = metrics.get("ece", "?")

    lines = [
        f"\n## {run_type} — {sport.upper()}",
        f"**Date:** {now}  |  **Run ID:** `{run_id[:8]}`",
        "",
        "| Metric   | Value  |",
        "|----------|--------|",
        f"| Log-loss | {ll:.4f} |" if isinstance(ll, float) else f"| Log-loss | {ll} |",
        f"| Accuracy | {acc:.2%} |" if isinstance(acc, float) else f"| Accuracy | {acc} |",
        f"| Brier    | {brier:.4f} |" if isinstance(brier, float) else f"| Brier    | {brier} |",
        f"| ECE      | {ece:.4f} |" if isinstance(ece, float) else f"| ECE      | {ece} |",
    ]
    if params:
        xgb = {k[4:]: v for k, v in params.items() if k.startswith("xgb_")}
        lgb = {k[4:]: v for k, v in params.items() if k.startswith("lgb_")}
        if xgb:
            lines += ["", f"**XGB params:** `{xgb}`"]
        if lgb:
            lines += [f"**LGB params:** `{lgb}`"]
    if notes:
        lines += ["", f"**Notes:** {notes}"]
    lines.append("\n---")

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _get_calibration_stats(sport: str) -> str:
    """Compute live log-loss on the active model's recent predictions."""
    try:
        from sklearn.metrics import log_loss
        from sqlalchemy import text

        with sync_session_factory() as session:
            rows = session.execute(
                text("""
                    SELECT p.probability, g.home_score, g.away_score
                    FROM predictions p
                    JOIN games g ON g.id = p.game_id
                    JOIN sports sp ON sp.id = g.sport_id
                    WHERE sp.code = :sport
                      AND g.status = 'final'
                      AND p.target = 'home_won'
                      AND g.home_score IS NOT NULL
                      AND g.scheduled_utc > NOW() - INTERVAL '180 days'
                """),
                {"sport": sport},
            ).fetchall()

        if not rows:
            return "n/a"
        probs = [float(r.probability) for r in rows]
        labels = [1 if r.home_score > r.away_score else 0 for r in rows]
        ll = log_loss(labels, probs)
        return f"{ll:.4f} (n={len(rows)})"
    except Exception:
        return "n/a"


def _notify_ops(message: str) -> None:
    from src.core.config import settings

    if settings.discord_webhook_ops:
        import contextlib

        import httpx

        with contextlib.suppress(Exception):
            httpx.post(settings.discord_webhook_ops, json={"content": message}, timeout=5)


def retrain_champion(sport: str, kind: str = "winner") -> dict:
    """Retrain the current champion using its exact hyperparameters on all new data.

    Same architecture, same feature set, fresh training data through yesterday.
    Auto-promotes if holdout logloss doesn't degrade by more than 0.005.
    This keeps the champion current without waiting for a weekly challenger gate.
    """
    import mlflow

    from src.core.config import settings
    from src.core.time import utc_now
    from src.models.registry import promote_model
    from src.models.train_winner import train_winner_model

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

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
        if champion is None:
            log.warning("retrain_champion.no_champion", sport=sport)
            return {"status": "no_champion"}

        # Pull the champion's exact hyperparameters from MLflow by searching for its run
        fixed_params = None
        try:
            _exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
            _exp_ids = [_exp.experiment_id] if _exp else ["0"]
            runs = client.search_runs(
                experiment_ids=_exp_ids,
                filter_string=(
                    f"tags.sport = '{sport}' AND tags.kind = '{kind}' AND status = 'FINISHED'"
                ),
                order_by=["metrics.logloss ASC"],
                max_results=50,
            )
            # Match the run whose ID starts with champion.version (8-char prefix)
            for r in runs:
                if r.info.run_id.startswith(champion.version):
                    fixed_params = dict(r.data.params)
                    break
        except Exception as exc:
            log.warning("retrain_champion.mlflow_lookup_failed", sport=sport, error=str(exc))

        champion_logloss = (champion.metrics or {}).get("logloss", 999.0)

    with sync_session_factory() as _load_session:
        df = _load_training_df(_load_session, sport)
    if len(df) < 100:
        return {"status": "skipped", "reason": "insufficient_data"}

    feature_names = _get_feature_names(sport, kind)
    max_season = df["season"].max()
    train_df = df[df["season"] < max_season].copy()
    holdout_df = df[df["season"] == max_season].copy()
    if train_df.empty or holdout_df.empty:
        cutoff = int(len(df) * 0.85)
        train_df = df.iloc[:cutoff].copy()
        holdout_df = df.iloc[cutoff:].copy()

    run_id, metrics = train_winner_model(
        sport=sport,
        training_df=train_df,
        feature_names=feature_names,
        holdout_df=holdout_df,
        n_optuna_trials=0,
        fixed_params=fixed_params,
        run_name=f"{sport}_{kind}_champion_refresh_{utc_now().strftime('%Y%m%d')}",
    )

    new_logloss = metrics.get("logloss", 999.0)
    degraded = new_logloss > champion_logloss + 0.005

    if degraded:
        log.warning(
            "retrain_champion.degraded",
            sport=sport,
            new_logloss=new_logloss,
            champion_logloss=champion_logloss,
        )
        _append_training_log(
            sport,
            "Champion Refresh ⚠️ (degraded — not promoted)",
            run_id,
            metrics,
            notes=f"Degraded: {champion_logloss:.4f} → {new_logloss:.4f}. Kept existing champion.",
        )
        _notify_ops(
            f"⚠️ {sport.upper()} champion refresh degraded "
            f"({new_logloss:.4f} vs {champion_logloss:.4f}) — keeping current champion."
        )
        return {
            "status": "degraded",
            "new_logloss": new_logloss,
            "champion_logloss": champion_logloss,
        }

    with sync_session_factory() as session:
        from src.features.common import feature_spec_hash

        sport_obj = session.query(SportModel).filter_by(code=sport).first()
        fs_hash = feature_spec_hash(feature_names)
        promote_model(
            session=session,
            run_id=run_id,
            sport_id=sport_obj.id,
            kind=kind,
            target="home_won",
            version=run_id[:8],
            metrics=metrics,
            feature_spec_hash=fs_hash,
        )
        session.commit()

    log.info("retrain_champion.promoted", sport=sport, logloss=new_logloss)
    _append_training_log(
        sport,
        "Champion Refresh ✅ (promoted)",
        run_id,
        metrics,
        notes=f"Same params, fresh data. LogLoss: {champion_logloss:.4f} → {new_logloss:.4f}",
    )
    _notify_ops(
        f"🔄 {sport.upper()} champion refreshed with new data. "
        f"LogLoss: {champion_logloss:.4f} → {new_logloss:.4f}"
    )
    return {"status": "promoted", "new_logloss": new_logloss, "run_id": run_id}


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
retrain_champion = app.task(name="src.tasks.train_tasks.retrain_champion", bind=False)(
    retrain_champion
)
