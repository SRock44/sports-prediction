"""Poll live MLB game state via GUMBO feed."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.db.models import Game, Sport
from src.ingest.mlb.client import get_live_feed

log = get_logger(__name__)


def update_live_scores(session: Session) -> dict[str, Any]:
    """Update scores and status for all active MLB games."""
    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        return {"updated": 0}

    # Fetch only games in 'scheduled' or 'in_progress' status today
    from datetime import timedelta

    from src.core.time import utc_now

    now = utc_now()
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=12)
    window_end = now + timedelta(hours=6)

    from sqlalchemy import or_

    active_games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= window_start,
            Game.scheduled_utc <= window_end,
            or_(
                Game.status.in_(["scheduled", "pre-game", "in_progress", "in progress", "warmup", "delayed", "delayed start"]),
                Game.status.like("in_progress_%"),
            ),
        )
        .all()
    )

    updated = 0
    for game in active_games:
        try:
            data = get_live_feed(int(game.external_id))
        except Exception as exc:
            log.warning("mlb.live.feed_failed", game_pk=game.external_id, error=str(exc))
            continue

        game_data = data.get("gameData", {})
        linescore = data.get("liveData", {}).get("linescore", {})
        status = game_data.get("status", {}).get("codedGameState", "")

        if status in ("F", "FR", "FT"):
            game.status = "final"
        elif status in ("I", "MA"):
            inning = linescore.get("currentInning", 1)
            game.status = f"in_progress_inning_{inning}"
        elif status in ("W", "PW"):
            game.status = "warmup"
        elif status in ("S",):
            game.status = "scheduled"

        home_score = linescore.get("teams", {}).get("home", {}).get("runs")
        away_score = linescore.get("teams", {}).get("away", {}).get("runs")
        if home_score is not None:
            game.home_score = int(home_score)
        if away_score is not None:
            game.away_score = int(away_score)

        updated += 1

    session.flush()
    log.info("mlb.live.updated", count=updated)
    return {"updated": updated}
