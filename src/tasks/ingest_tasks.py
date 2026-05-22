"""Celery ingest tasks: daily box scores, injury refresh, live polling."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from celery import shared_task

from src.core.logging import get_logger
from src.db.session import sync_session_factory
from src.ingest.common import IngestResult

log = get_logger(__name__)


@shared_task(name="src.tasks.ingest_tasks.ingest_yesterday_nba", bind=True, max_retries=3)
def ingest_yesterday_nba(self: Any) -> dict:
    """Ingest box scores only for games completed yesterday — not the whole season."""
    from src.db.models import Game, Sport
    from src.ingest.nba.games import ingest_box_scores

    yesterday = date.today() - timedelta(days=1)
    day_start = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)

    total = IngestResult()
    with sync_session_factory() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            return {"inserted": 0, "updated": 0}

        games = (
            session.query(Game)
            .filter(
                Game.sport_id == sport.id,
                Game.scheduled_utc >= day_start,
                Game.scheduled_utc < day_end,
                Game.status == "final",
            )
            .all()
        )

        for game in games:
            r = ingest_box_scores(session, game.external_id)
            total += r
        session.commit()

    log.info("task.nba_ingest.done", inserted=total.rows_inserted, updated=total.rows_updated)
    return {"inserted": total.rows_inserted, "updated": total.rows_updated}


@shared_task(name="src.tasks.ingest_tasks.ingest_yesterday_mlb", bind=True, max_retries=3)
def ingest_yesterday_mlb(self: Any) -> dict:
    """Ingest box scores only for games completed yesterday — not the whole season."""
    from src.db.models import Game, Sport
    from src.ingest.mlb.games import ingest_box_score

    yesterday = date.today() - timedelta(days=1)
    day_start = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)

    total = IngestResult()
    with sync_session_factory() as session:
        sport = session.query(Sport).filter_by(code="mlb").first()
        if sport is None:
            return {"inserted": 0, "updated": 0}

        games = (
            session.query(Game)
            .filter(
                Game.sport_id == sport.id,
                Game.scheduled_utc >= day_start,
                Game.scheduled_utc < day_end,
                Game.status == "final",
            )
            .all()
        )

        for game in games:
            r = ingest_box_score(session, game.external_id)
            total += r
        session.commit()

    log.info("task.mlb_ingest.done", inserted=total.rows_inserted)
    return {"inserted": total.rows_inserted, "updated": total.rows_updated}


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


@shared_task(name="src.tasks.ingest_tasks.ingest_odds_open", bind=True, max_retries=2)
def ingest_odds_open(self: Any) -> dict:
    """Fetch opening lines from DraftKings, FanDuel, and Kalshi (~24h before tip-off)."""
    from src.ingest.odds.games import ingest_odds

    total = 0
    with sync_session_factory() as session:
        for sport in ("nba", "mlb"):
            r = ingest_odds(session, sport, snapshot="open")
            total += r.rows_inserted
        session.commit()
    return {"inserted": total}


@shared_task(name="src.tasks.ingest_tasks.ingest_odds_close", bind=True, max_retries=2)
def ingest_odds_close(self: Any) -> dict:
    """Fetch closing lines from DraftKings, FanDuel, and Kalshi (~1h before tip-off)."""
    from src.ingest.odds.games import ingest_odds

    total = 0
    with sync_session_factory() as session:
        for sport in ("nba", "mlb"):
            r = ingest_odds(session, sport, snapshot="close")
            total += r.rows_inserted
        session.commit()
    return {"inserted": total}


@shared_task(name="src.tasks.ingest_tasks.ingest_mlb_weather", bind=True, max_retries=2)
def ingest_mlb_weather(self: Any) -> dict:
    """Fetch weather forecasts for upcoming MLB outdoor games."""
    from src.ingest.mlb.weather import ingest_weather_for_upcoming

    with sync_session_factory() as session:
        result = ingest_weather_for_upcoming(session, lookahead_days=5)
        session.commit()
    return {"inserted": result.rows_inserted, "updated": result.rows_updated}


from typing import Any  # noqa: E402
