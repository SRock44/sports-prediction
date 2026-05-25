"""Celery beat schedule — single source of truth for all periodic tasks.

All times are UTC. EST = UTC-5.
"""

from __future__ import annotations

from celery.schedules import crontab

BEAT_SCHEDULE = {
    # ── Daily schedule refresh (3:00 AM EST / 8:00 AM UTC) ──────────────────────
    # Runs before the 4 AM ingest chain so upcoming games exist when scoring fires.
    "ingest-schedule-nba": {
        "task": "src.tasks.ingest_tasks.ingest_schedule_nba",
        "schedule": crontab(hour=8, minute=0),  # 3:00 AM EST
        "kwargs": {"days_ahead": 7},
    },
    "ingest-schedule-mlb": {
        "task": "src.tasks.ingest_tasks.ingest_schedule_mlb",
        "schedule": crontab(hour=8, minute=0),  # 3:00 AM EST (parallel)
    },
    # ── Daily ingest (runs twice for NBA to catch late/playoff games past midnight UTC)
    # First pass: 1:30 AM EST / 6:30 AM UTC — catches games ending ~11 PM-1 AM ET
    # Second pass: 4:00 AM EST / 9:00 AM UTC — catches any stragglers + MLB
    "ingest-nba-yesterday-early": {
        "task": "src.tasks.ingest_tasks.ingest_yesterday_nba",
        "schedule": crontab(hour=6, minute=30),  # 1:30 AM EST
    },
    "ingest-nba-yesterday": {
        "task": "src.tasks.ingest_tasks.ingest_yesterday_nba",
        "schedule": crontab(hour=9, minute=0),  # 4:00 AM EST
    },
    "ingest-mlb-yesterday": {
        "task": "src.tasks.ingest_tasks.ingest_yesterday_mlb",
        "schedule": crontab(hour=9, minute=0),  # 4:00 AM EST (parallel)
    },
    # ── MLB SP feature patch (4:30 AM EST / 9:30 AM UTC — after box scores ingest) ─
    "patch-sp-features-mlb": {
        "task": "src.tasks.ingest_tasks.patch_sp_features_mlb",
        "schedule": crontab(hour=9, minute=30),  # 4:30 AM EST
    },
    # ── Injury reports (6:00 AM EST / 11:00 AM UTC) ───────────────────────────
    "refresh-nba-injuries": {
        "task": "src.tasks.ingest_tasks.refresh_nba_injuries",
        "schedule": crontab(hour=11, minute=0),  # 6:00 AM EST
    },
    "refresh-mlb-il": {
        "task": "src.tasks.ingest_tasks.refresh_mlb_il",
        "schedule": crontab(hour=11, minute=0),  # 6:00 AM EST (parallel)
    },
    # ── Feature rebuild (7:00 AM EST / 12:00 PM UTC) ─────────────────────────
    "rebuild-features-nba": {
        "task": "src.tasks.feature_tasks.rebuild_features_nba",
        "schedule": crontab(hour=12, minute=0),  # 7:00 AM EST
    },
    "rebuild-features-mlb": {
        "task": "src.tasks.feature_tasks.rebuild_features_mlb",
        "schedule": crontab(hour=12, minute=0),  # 7:00 AM EST (parallel)
    },
    # ── Challenger + champion training (7:40 AM EST / 12:40 PM UTC) ──────────
    # Both sports run in parallel. Champion finishes in ~5s; challenger ~90 min.
    # Weekly promotion gate (Mon 11 PM EST) decides if challenger beats champion.
    "train-challenger-nba-winner": {
        "task": "src.tasks.train_tasks.train_challenger",
        "schedule": crontab(hour=12, minute=40),  # 7:40 AM EST
        "kwargs": {"sport": "nba", "kind": "winner"},
    },
    "train-challenger-mlb-winner": {
        "task": "src.tasks.train_tasks.train_challenger",
        "schedule": crontab(hour=12, minute=40),  # 7:40 AM EST (parallel)
        "kwargs": {"sport": "mlb", "kind": "winner"},
    },
    "retrain-champion-nba": {
        "task": "src.tasks.train_tasks.retrain_champion",
        "schedule": crontab(hour=12, minute=45),  # 7:45 AM EST
        "kwargs": {"sport": "nba", "kind": "winner"},
    },
    "retrain-champion-mlb": {
        "task": "src.tasks.train_tasks.retrain_champion",
        "schedule": crontab(hour=12, minute=45),  # 7:45 AM EST (parallel)
        "kwargs": {"sport": "mlb", "kind": "winner"},
    },
    # ── Props model training (7:55 AM EST / 12:55 PM UTC — after champion refresh) ─
    "train-props-nba": {
        "task": "src.tasks.train_tasks.train_props",
        "schedule": crontab(hour=12, minute=55),  # 7:55 AM EST
        "kwargs": {"sport": "nba"},
    },
    "train-props-mlb": {
        "task": "src.tasks.train_tasks.train_props",
        "schedule": crontab(hour=12, minute=55),  # 7:55 AM EST (parallel)
        "kwargs": {"sport": "mlb"},
    },
    # ── Score upcoming games (8:00 AM EST / 1:00 PM UTC) ─────────────────────
    # Runs after champion refresh — picks use today's fresh champion model.
    "score-nba": {
        "task": "src.tasks.score_tasks.score_nba_upcoming",
        "schedule": crontab(hour=13, minute=0),  # 8:00 AM EST
    },
    "score-mlb": {
        "task": "src.tasks.score_tasks.score_mlb_upcoming",
        "schedule": crontab(hour=13, minute=0),  # 8:00 AM EST (parallel)
    },
    # ── Props scoring (8:15 AM EST / 1:15 PM UTC — after game scoring + DK lines open) ─
    # Also every 30 min 10 AM-5 PM EST as lineups/lines update
    "score-props-nba": {
        "task": "src.tasks.score_tasks.score_props_upcoming_nba",
        "schedule": crontab(hour=13, minute=15),  # 8:15 AM EST
    },
    "score-props-mlb": {
        "task": "src.tasks.score_tasks.score_props_upcoming_mlb",
        "schedule": crontab(hour=13, minute=15),  # 8:15 AM EST (parallel)
    },
    "rescore-props-nba": {
        "task": "src.tasks.score_tasks.score_props_upcoming_nba",
        "schedule": crontab(minute="*/30", hour="15-22"),  # 10 AM-5 PM EST
    },
    "rescore-props-mlb": {
        "task": "src.tasks.score_tasks.score_props_upcoming_mlb",
        "schedule": crontab(minute="*/30", hour="17-23"),  # 12 PM-6 PM EST
    },
    # ── Daily top props post (8:30 AM EST / 1:30 PM UTC — after scoring + lines open) ──
    "post-daily-props-nba": {
        "task": "src.tasks.outcome_tasks.post_daily_props",
        "schedule": crontab(hour=13, minute=30),  # 8:30 AM EST
        "kwargs": {"sport": "nba"},
    },
    "post-daily-props-mlb": {
        "task": "src.tasks.outcome_tasks.post_daily_props",
        "schedule": crontab(hour=14, minute=0),  # 9:00 AM EST (same time as game picks)
        "kwargs": {"sport": "mlb"},
    },
    # ── Odds (opening lines 2:00 PM UTC / 9:00 AM EST, close 10:00 PM UTC / 5:00 PM EST)
    "ingest-odds-open": {
        "task": "src.tasks.ingest_tasks.ingest_odds_open",
        "schedule": crontab(hour=14, minute=0),  # 9:00 AM EST
    },
    "ingest-odds-close": {
        "task": "src.tasks.ingest_tasks.ingest_odds_close",
        "schedule": crontab(hour=22, minute=0),  # 5:00 PM EST
    },
    # ── Daily top-10 picks post (2:00 PM UTC / 9:00 AM EST) ──────────────────
    "post-daily-picks-nba": {
        "task": "src.tasks.outcome_tasks.post_daily_picks",
        "schedule": crontab(hour=14, minute=0),  # 9:00 AM EST
        "kwargs": {"sport": "nba"},
    },
    "post-daily-picks-mlb": {
        "task": "src.tasks.outcome_tasks.post_daily_picks",
        "schedule": crontab(hour=14, minute=30),  # 9:30 AM EST
        "kwargs": {"sport": "mlb"},
    },
    # ── Re-score on lineup confirmation (every 15 min 10:00 AM-5:00 PM EST) ──
    "score-nba-lineup-update": {
        "task": "src.tasks.score_tasks.rescore_on_lineup_change_nba",
        "schedule": crontab(minute="*/15", hour="15-22"),  # 10:00 AM-5:00 PM EST
    },
    "score-mlb-lineup-update": {
        "task": "src.tasks.score_tasks.rescore_on_lineup_change_mlb",
        "schedule": crontab(minute="*/15", hour="17-23"),  # 12:00 PM-6:00 PM EST
    },
    # ── MLB weather (6:00 PM UTC / 1:00 PM EST — 5-day lookahead) ────────────
    "ingest-mlb-weather": {
        "task": "src.tasks.ingest_tasks.ingest_mlb_weather",
        "schedule": crontab(hour=18, minute=0),  # 1:00 PM EST
    },
    # ── Live polling (every 2 min — lightweight score updates) ───────────────
    "live-nba": {
        "task": "src.tasks.ingest_tasks.poll_live_nba",
        "schedule": crontab(minute="*/2"),
    },
    "live-mlb": {
        "task": "src.tasks.ingest_tasks.poll_live_mlb",
        "schedule": crontab(minute="*/2"),
    },
    # ── Outcome checks (every 5 min 1:00 AM-12:00 AM EST — game windows) ─────
    "check-outcomes-nba": {
        "task": "src.tasks.outcome_tasks.check_outcomes_nba",
        "schedule": crontab(minute="*/5", hour="18-23,0-5"),  # 1:00 PM-12:00 AM EST
    },
    "check-outcomes-mlb": {
        "task": "src.tasks.outcome_tasks.check_outcomes_mlb",
        "schedule": crontab(minute="*/5", hour="18-23,0-5"),  # 1:00 PM-12:00 AM EST
    },
    # ── Drift detection (11:00/11:30 PM EST / 4:00/4:30 AM UTC) ─────────────
    "drift-monitor-nba": {
        "task": "src.tasks.train_tasks.run_drift_monitor",
        "schedule": crontab(hour=4, minute=0),  # 11:00 PM EST
        "kwargs": {"sport": "nba"},
    },
    "drift-monitor-mlb": {
        "task": "src.tasks.train_tasks.run_drift_monitor",
        "schedule": crontab(hour=4, minute=30),  # 11:30 PM EST
        "kwargs": {"sport": "mlb"},
    },
    # ── Daily promotion gate (1:00/1:15 AM EST / 6:00/6:15 UTC) ────────────
    # After west coast late games finish (~12:45 AM EST). Challenger trained
    # on freshest data each morning; best log-loss wins every night.
    "evaluate-and-promote-nba": {
        "task": "src.tasks.train_tasks.evaluate_and_promote",
        "schedule": crontab(hour=6, minute=0),  # 1:00 AM EST
        "kwargs": {"sport": "nba"},
    },
    "evaluate-and-promote-mlb": {
        "task": "src.tasks.train_tasks.evaluate_and_promote",
        "schedule": crontab(hour=6, minute=15),  # 1:15 AM EST
        "kwargs": {"sport": "mlb"},
    },
    # ── Weekly backtest report (Sat 10:00/10:30 PM EST / Sun 3:00/3:30 AM UTC)
    "backtest-report-nba": {
        "task": "src.tasks.train_tasks.generate_backtest_report",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),  # Sat 10:00 PM EST
        "kwargs": {"sport": "nba"},
    },
    "backtest-report-mlb": {
        "task": "src.tasks.train_tasks.generate_backtest_report",
        "schedule": crontab(hour=3, minute=30, day_of_week=0),  # Sat 10:30 PM EST
        "kwargs": {"sport": "mlb"},
    },
    # ── Monthly hyperparameter search (1st of month, 12:00/12:30 AM EST / 5:00/5:30 AM UTC)
    "hyperparam-search-nba": {
        "task": "src.tasks.train_tasks.hyperparam_search",
        "schedule": crontab(hour=5, minute=0, day_of_month=1),  # 12:00 AM EST
        "kwargs": {"sport": "nba"},
    },
    "hyperparam-search-mlb": {
        "task": "src.tasks.train_tasks.hyperparam_search",
        "schedule": crontab(hour=5, minute=30, day_of_month=1),  # 12:30 AM EST
        "kwargs": {"sport": "mlb"},
    },
}
