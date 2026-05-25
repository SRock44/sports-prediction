"""Score upcoming games."""

from __future__ import annotations

from src.core.logging import get_logger
from src.db.session import sync_session_factory

log = get_logger(__name__)


def _score_sport(sport: str) -> dict:
    from src.models.score import score_upcoming_games

    with sync_session_factory() as session:
        count = score_upcoming_games(session, sport)
        session.commit()
    return {"sport": sport, "predictions_written": count}


def score_nba_upcoming() -> dict:
    return _score_sport("nba")


def score_mlb_upcoming() -> dict:
    return _score_sport("mlb")


def rescore_on_lineup_change_nba() -> dict:
    """Re-score NBA games where lineups changed since last scoring."""
    from src.models.score import score_upcoming_games

    with sync_session_factory() as session:
        count = score_upcoming_games(session, "nba", hours_ahead=8)
        session.commit()
    return {"predictions_written": count}


def rescore_on_lineup_change_mlb() -> dict:
    from src.models.score import score_upcoming_games

    with sync_session_factory() as session:
        count = score_upcoming_games(session, "mlb", hours_ahead=8)
        session.commit()
    return {"predictions_written": count}


def score_props_upcoming_nba() -> dict:
    from src.models.score import score_props_upcoming

    with sync_session_factory() as session:
        count = score_props_upcoming(session, "nba")
        session.commit()
    return {"sport": "nba", "predictions_written": count}


def score_props_upcoming_mlb() -> dict:
    from src.models.score import score_props_upcoming

    with sync_session_factory() as session:
        count = score_props_upcoming(session, "mlb")
        session.commit()
    return {"sport": "mlb", "predictions_written": count}


# Register as Celery tasks
from src.tasks.celery_app import app  # noqa: E402

score_nba_upcoming = app.task(name="src.tasks.score_tasks.score_nba_upcoming", bind=False)(
    score_nba_upcoming
)
score_mlb_upcoming = app.task(name="src.tasks.score_tasks.score_mlb_upcoming", bind=False)(
    score_mlb_upcoming
)
rescore_on_lineup_change_nba = app.task(
    name="src.tasks.score_tasks.rescore_on_lineup_change_nba", bind=False
)(rescore_on_lineup_change_nba)
rescore_on_lineup_change_mlb = app.task(
    name="src.tasks.score_tasks.rescore_on_lineup_change_mlb", bind=False
)(rescore_on_lineup_change_mlb)
score_props_upcoming_nba = app.task(
    name="src.tasks.score_tasks.score_props_upcoming_nba", bind=False
)(score_props_upcoming_nba)
score_props_upcoming_mlb = app.task(
    name="src.tasks.score_tasks.score_props_upcoming_mlb", bind=False
)(score_props_upcoming_mlb)
