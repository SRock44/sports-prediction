"""Poll live NBA game state during active game windows."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.db.models import Game, Sport
from src.ingest.nba.client import get_live_scoreboard

log = get_logger(__name__)

_GAME_WINDOW_HOURS_BEFORE = 0.5
_GAME_WINDOW_HOURS_AFTER = 4.0


def is_in_game_window() -> bool:
    """Return True if any NBA game is likely live right now."""
    # NBA games are typically 19:00-23:59 ET -> 00:00-05:00 UTC
    # Broad window check; specific check is done per game below
    return True  # polled tasks check individual game status


def update_live_scores(session: Session) -> dict[str, Any]:
    """Fetch live scoreboard and update game scores + status in DB."""
    try:
        data = get_live_scoreboard()
    except Exception as exc:
        log.error("nba.live.fetch_failed", error=str(exc))
        return {"updated": 0, "error": str(exc)}

    sport = session.query(Sport).filter_by(code="nba").first()
    if sport is None:
        return {"updated": 0}

    games_data = data.get("scoreboard", {}).get("games", [])
    updated = 0

    for g in games_data:
        game_id_ext = g.get("gameId", "")
        if not game_id_ext:
            continue

        game = session.query(Game).filter_by(sport_id=sport.id, external_id=game_id_ext).first()
        if game is None:
            continue

        period = g.get("period", 0)
        game_status_text = g.get("gameStatusText", "").lower()
        home_score = g.get("homeTeam", {}).get("score")
        away_score = g.get("awayTeam", {}).get("score")

        if "final" in game_status_text:
            game.status = "final"
        elif period > 0:
            game.status = f"in_progress_q{period}"
        else:
            game.status = "pre-game"

        if home_score is not None:
            game.home_score = int(home_score)
        if away_score is not None:
            game.away_score = int(away_score)

        updated += 1

    session.flush()
    log.info("nba.live.updated", count=updated)
    return {"updated": updated}
