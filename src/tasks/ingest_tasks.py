"""Celery ingest tasks: daily box scores, injury refresh, live polling."""
from __future__ import annotations

from datetime import date, timedelta

from celery import shared_task

from src.core.logging import get_logger
from src.db.session import sync_session_factory

log = get_logger(__name__)


@shared_task(name="src.tasks.ingest_tasks.ingest_yesterday_nba", bind=True, max_retries=3)
def ingest_yesterday_nba(self: Any) -> dict:
    from src.ingest.nba.games import ingest_season_schedule, ingest_box_scores
    from src.core.time import nba_season_for_date
    yesterday = date.today() - timedelta(days=1)
    season = nba_season_for_date(yesterday)

    with sync_session_factory() as session:
        result = ingest_season_schedule(session, season)
        session.commit()
    log.info("task.nba_ingest.done", inserted=result.rows_inserted, updated=result.rows_updated)
    return {"inserted": result.rows_inserted, "updated": result.rows_updated}


@shared_task(name="src.tasks.ingest_tasks.ingest_yesterday_mlb", bind=True, max_retries=3)
def ingest_yesterday_mlb(self: Any) -> dict:
    from src.ingest.mlb.games import ingest_season_schedule
    yesterday = date.today() - timedelta(days=1)
    season = yesterday.year

    with sync_session_factory() as session:
        result = ingest_season_schedule(session, season)
        session.commit()
    log.info("task.mlb_ingest.done", inserted=result.rows_inserted)
    return {"inserted": result.rows_inserted}


@shared_task(name="src.tasks.ingest_tasks.refresh_nba_injuries", bind=True, max_retries=2)
def refresh_nba_injuries(self: Any) -> dict:
    from src.ingest.nba.players import ingest_injury_report
    with sync_session_factory() as session:
        result = ingest_injury_report(session)
        session.commit()
    return {"inserted": result.rows_inserted}


@shared_task(name="src.tasks.ingest_tasks.refresh_mlb_il", bind=True, max_retries=2)
def refresh_mlb_il(self: Any) -> dict:
    from src.ingest.mlb.players import ingest_il_transactions
    with sync_session_factory() as session:
        result = ingest_il_transactions(session, lookback_days=2)
        session.commit()
    return {"inserted": result.rows_inserted}


@shared_task(name="src.tasks.ingest_tasks.poll_live_nba")
def poll_live_nba() -> dict:
    from src.ingest.nba.live import update_live_scores
    with sync_session_factory() as session:
        result = update_live_scores(session)
        session.commit()
    return result


@shared_task(name="src.tasks.ingest_tasks.poll_live_mlb")
def poll_live_mlb() -> dict:
    from src.ingest.mlb.live import update_live_scores
    with sync_session_factory() as session:
        result = update_live_scores(session)
        session.commit()
    return result


from typing import Any  # noqa: E402
