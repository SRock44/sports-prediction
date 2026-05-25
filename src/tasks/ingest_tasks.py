"""Celery ingest tasks: daily box scores, injury refresh, live polling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from celery import shared_task

from src.core.logging import get_logger
from src.db.session import sync_session_factory
from src.ingest.common import IngestResult

log = get_logger(__name__)

_ET = ZoneInfo("America/New_York")


def _yesterday_et_window() -> tuple[datetime, datetime]:
    """Return (start, end) UTC datetimes covering yesterday in ET.

    Uses ET midnight boundaries so late games (8-11 PM ET) that roll past
    UTC midnight are still included in yesterday's ingest window.
    """
    now_et = datetime.now(_ET)
    yesterday_et = (now_et - timedelta(days=1)).date()
    day_start = datetime(yesterday_et.year, yesterday_et.month, yesterday_et.day, tzinfo=_ET)
    return day_start.astimezone(UTC), (day_start + timedelta(days=1)).astimezone(UTC)


@shared_task(name="src.tasks.ingest_tasks.ingest_yesterday_nba", bind=True, max_retries=3)
def ingest_yesterday_nba(self: Any) -> dict:
    """Ingest box scores for games from yesterday ET (ET-midnight boundaries)."""
    from src.db.models import Game, Sport
    from src.ingest.nba.games import ingest_box_scores

    day_start, day_end = _yesterday_et_window()
    now_utc = datetime.now(UTC)

    total = IngestResult()
    with sync_session_factory() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            return {"inserted": 0, "updated": 0}

        # Include any game whose tip-off was yesterday ET and has already started —
        # even if the live poller hasn't flipped it to "final" yet.
        games = (
            session.query(Game)
            .filter(
                Game.sport_id == sport.id,
                Game.scheduled_utc >= day_start,
                Game.scheduled_utc < day_end,
                Game.scheduled_utc < now_utc,
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
    """Ingest box scores for games from yesterday ET (ET-midnight boundaries)."""
    from src.db.models import Game, Sport
    from src.ingest.mlb.games import ingest_box_score

    day_start, day_end = _yesterday_et_window()
    now_utc = datetime.now(UTC)

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
                Game.scheduled_utc < now_utc,
            )
            .all()
        )

        for game in games:
            r = ingest_box_score(session, game.external_id)
            total += r
        session.commit()

    log.info("task.mlb_ingest.done", inserted=total.rows_inserted)
    return {"inserted": total.rows_inserted, "updated": total.rows_updated}


@shared_task(name="src.tasks.ingest_tasks.patch_sp_features_mlb", bind=True, max_retries=2)
def patch_sp_features_mlb(self: Any) -> dict:
    """Backfill SP form features into matchup_features for recently completed MLB games.

    Runs after ingest_yesterday_mlb so box scores are available. Patches only
    games where home_sp_form_known=0 — idempotent.
    """
    import json

    from sqlalchemy import text

    from src.core.time import as_of_for_game
    from src.features.mlb.matchup import _get_confirmed_starter, _sp_rolling_form

    patched = skipped = 0
    with sync_session_factory() as session:
        rows = session.execute(
            text("""
                SELECT g.id, g.home_team_id, g.away_team_id, g.scheduled_utc
                FROM games g
                JOIN sports sp ON sp.id = g.sport_id
                JOIN matchup_features mf ON mf.game_id = g.id
                WHERE sp.code = 'mlb'
                  AND g.status = 'final'
                  AND g.scheduled_utc > NOW() - INTERVAL '3 days'
                  AND (mf.features->>'home_sp_form_known' IS NULL
                       OR (mf.features->>'home_sp_form_known')::int = 0)
            """)
        ).fetchall()

        for row in rows:
            try:
                as_of = as_of_for_game(row.scheduled_utc)
                home_sp = _get_confirmed_starter(session, row.id, row.home_team_id, as_of)
                away_sp = _get_confirmed_starter(session, row.id, row.away_team_id, as_of)
                home_sp_id = (
                    (home_sp.get("playerId") or home_sp.get("player_id")) if home_sp else None
                )
                away_sp_id = (
                    (away_sp.get("playerId") or away_sp.get("player_id")) if away_sp else None
                )

                if home_sp_id is None and away_sp_id is None:
                    skipped += 1
                    continue

                home_form = _sp_rolling_form(session, home_sp_id, as_of, prefix="home_sp")
                away_form = _sp_rolling_form(session, away_sp_id, as_of, prefix="away_sp")

                if not home_form.get("home_sp_form_known") and not away_form.get(
                    "away_sp_form_known"
                ):
                    skipped += 1
                    continue

                patch = {
                    **home_form,
                    **away_form,
                    "sp_form_era_diff": home_form.get("home_sp_form_era", 4.50)
                    - away_form.get("away_sp_form_era", 4.50),
                    "sp_form_k_pct_diff": home_form.get("home_sp_form_k_pct", 0.22)
                    - away_form.get("away_sp_form_k_pct", 0.22),
                }
                session.execute(
                    text(
                        "UPDATE matchup_features SET features = features || CAST(:patch AS jsonb) WHERE game_id = :gid"
                    ),
                    {"gid": row.id, "patch": json.dumps(patch)},
                )
                patched += 1
            except Exception as exc:
                log.warning("patch_sp.error", game_id=row.id, error=str(exc))
                skipped += 1

        session.commit()

    log.info("patch_sp_features.done", patched=patched, skipped=skipped)
    return {"patched": patched, "skipped": skipped}


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


@shared_task(name="src.tasks.ingest_tasks.ingest_schedule_nba", bind=True, max_retries=2)
def ingest_schedule_nba(self: Any, days_ahead: int = 7) -> dict:
    """Upsert upcoming NBA games (ScoreboardV2 for each of the next N days)."""
    from src.ingest.nba.games import ingest_upcoming_nba_schedule

    with sync_session_factory() as session:
        result = ingest_upcoming_nba_schedule(session, days_ahead=days_ahead)
        session.commit()
    return {"inserted": result.rows_inserted, "updated": result.rows_updated}


@shared_task(name="src.tasks.ingest_tasks.ingest_schedule_mlb", bind=True, max_retries=2)
def ingest_schedule_mlb(self: Any, days_ahead: int = 7) -> dict:
    """Upsert upcoming MLB games (re-fetches the rest of the current season schedule)."""
    from datetime import date

    from src.core.time import mlb_season_for_date
    from src.ingest.mlb.games import ingest_season_schedule

    season = mlb_season_for_date(date.today())
    with sync_session_factory() as session:
        result = ingest_season_schedule(session, season)
        session.commit()
    return {"inserted": result.rows_inserted, "updated": result.rows_updated}


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
