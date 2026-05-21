"""Celery beat schedule — single source of truth for all periodic tasks."""
from __future__ import annotations

from celery.schedules import crontab

BEAT_SCHEDULE = {
    # ── Daily ingest (09:00 UTC — box scores finalized by then) ───────────────
    "ingest-nba-yesterday": {
        "task": "src.tasks.ingest_tasks.ingest_yesterday_nba",
        "schedule": crontab(hour=9, minute=0),
    },
    "ingest-mlb-yesterday": {
        "task": "src.tasks.ingest_tasks.ingest_yesterday_mlb",
        "schedule": crontab(hour=9, minute=30),
    },
    # ── Injury reports (11:00 UTC) ────────────────────────────────────────────
    "refresh-nba-injuries": {
        "task": "src.tasks.ingest_tasks.refresh_nba_injuries",
        "schedule": crontab(hour=11, minute=0),
    },
    "refresh-mlb-il": {
        "task": "src.tasks.ingest_tasks.refresh_mlb_il",
        "schedule": crontab(hour=11, minute=30),
    },
    # ── Feature rebuild (12:00 UTC) ───────────────────────────────────────────
    "rebuild-features-nba": {
        "task": "src.tasks.feature_tasks.rebuild_features_nba",
        "schedule": crontab(hour=12, minute=0),
    },
    "rebuild-features-mlb": {
        "task": "src.tasks.feature_tasks.rebuild_features_mlb",
        "schedule": crontab(hour=12, minute=30),
    },
    # ── Score upcoming games (13:00 UTC) ─────────────────────────────────────
    "score-nba": {
        "task": "src.tasks.score_tasks.score_nba_upcoming",
        "schedule": crontab(hour=13, minute=0),
    },
    "score-mlb": {
        "task": "src.tasks.score_tasks.score_mlb_upcoming",
        "schedule": crontab(hour=13, minute=30),
    },
    # ── Re-score on lineup confirmation (every 15 min 15:00–22:00 UTC) ───────
    "score-nba-lineup-update": {
        "task": "src.tasks.score_tasks.rescore_on_lineup_change_nba",
        "schedule": crontab(minute="*/15", hour="15-22"),
    },
    "score-mlb-lineup-update": {
        "task": "src.tasks.score_tasks.rescore_on_lineup_change_mlb",
        "schedule": crontab(minute="*/15", hour="17-23"),
    },
    # ── Live polling (every 2 min during game windows — lightweight) ──────────
    "live-nba": {
        "task": "src.tasks.ingest_tasks.poll_live_nba",
        "schedule": crontab(minute="*/2"),
    },
    "live-mlb": {
        "task": "src.tasks.ingest_tasks.poll_live_mlb",
        "schedule": crontab(minute="*/2"),
    },
    # ── Nightly training (02:00 UTC — challenger model) ───────────────────────
    "train-challenger-nba-winner": {
        "task": "src.tasks.train_tasks.train_challenger",
        "schedule": crontab(hour=2, minute=0),
        "kwargs": {"sport": "nba", "kind": "winner"},
    },
    "train-challenger-mlb-winner": {
        "task": "src.tasks.train_tasks.train_challenger",
        "schedule": crontab(hour=2, minute=30),
        "kwargs": {"sport": "mlb", "kind": "winner"},
    },
    # ── Drift detection (03:00 UTC) ───────────────────────────────────────────
    "drift-monitor-nba": {
        "task": "src.tasks.train_tasks.run_drift_monitor",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"sport": "nba"},
    },
    "drift-monitor-mlb": {
        "task": "src.tasks.train_tasks.run_drift_monitor",
        "schedule": crontab(hour=3, minute=30),
        "kwargs": {"sport": "mlb"},
    },
    # ── Weekly promotion gate (Mon 04:00 UTC) ─────────────────────────────────
    "evaluate-and-promote-nba": {
        "task": "src.tasks.train_tasks.evaluate_and_promote",
        "schedule": crontab(hour=4, minute=0, day_of_week=1),
        "kwargs": {"sport": "nba"},
    },
    "evaluate-and-promote-mlb": {
        "task": "src.tasks.train_tasks.evaluate_and_promote",
        "schedule": crontab(hour=4, minute=30, day_of_week=1),
        "kwargs": {"sport": "mlb"},
    },
    # ── Weekly backtest report (Sun 03:00 UTC) ────────────────────────────────
    "backtest-report-nba": {
        "task": "src.tasks.train_tasks.generate_backtest_report",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),
        "kwargs": {"sport": "nba"},
    },
    "backtest-report-mlb": {
        "task": "src.tasks.train_tasks.generate_backtest_report",
        "schedule": crontab(hour=3, minute=30, day_of_week=0),
        "kwargs": {"sport": "mlb"},
    },
    # ── Monthly hyperparameter search (1st of month 05:00 UTC) ───────────────
    "hyperparam-search-nba": {
        "task": "src.tasks.train_tasks.hyperparam_search",
        "schedule": crontab(hour=5, minute=0, day_of_month=1),
        "kwargs": {"sport": "nba"},
    },
    "hyperparam-search-mlb": {
        "task": "src.tasks.train_tasks.hyperparam_search",
        "schedule": crontab(hour=5, minute=30, day_of_month=1),
        "kwargs": {"sport": "mlb"},
    },
}
