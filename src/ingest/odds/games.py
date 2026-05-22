"""Fetch and store opening/closing lines from DraftKings, FanDuel, and Kalshi."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.db.models import Game, Sport
from src.db.models.odds import GameOdds
from src.ingest.common import IngestResult
from src.ingest.odds.client import get_consensus_lines, get_kalshi_probs

log = get_logger(__name__)


def ingest_odds(
    session: Session,
    sport_code: str,
    snapshot: str = "open",
) -> IngestResult:
    """Fetch current lines from DK + FD + Kalshi and store against DB games.

    Args:
        snapshot: 'open' (first fetch, ~24h before game) or 'close' (~1h before).
    """
    result = IngestResult()
    now = datetime.now(timezone.utc)

    sport = session.query(Sport).filter_by(code=sport_code.lower()).first()
    if sport is None:
        result.errors.append(f"Sport {sport_code} not found")
        return result

    # ── DraftKings + FanDuel consensus lines ──────────────────────────────────
    try:
        consensus_games = get_consensus_lines(sport_code)
    except Exception as exc:
        log.error("odds.consensus_failed", sport=sport_code, error=str(exc))
        consensus_games = []
        result.errors.append(str(exc))

    for game_data in consensus_games:
        game = _match_game(session, sport, game_data)
        if game is None:
            continue

        # Store DraftKings line
        if game_data.get("dk_home_ml") is not None:
            _upsert(session, game.id, "draftkings", "h2h", snapshot,
                    game_data["dk_home_ml"], game_data.get("dk_away_ml"), None, now)
            result.rows_inserted += 1
        if game_data.get("dk_home_spread") is not None:
            _upsert(session, game.id, "draftkings", "spreads", snapshot,
                    None, None, game_data["dk_home_spread"], now)
            result.rows_inserted += 1

        # Store FanDuel line
        if game_data.get("fd_home_ml") is not None:
            _upsert(session, game.id, "fanduel", "h2h", snapshot,
                    game_data["fd_home_ml"], game_data.get("fd_away_ml"), None, now)
            result.rows_inserted += 1
        if game_data.get("fd_home_spread") is not None:
            _upsert(session, game.id, "fanduel", "spreads", snapshot,
                    None, None, game_data["fd_home_spread"], now)
            result.rows_inserted += 1

        # Store consensus h2h (average of available books)
        if game_data.get("consensus_home_implied_prob") != 0.5:
            _upsert(session, game.id, "consensus", "h2h", snapshot,
                    _prob_to_american(game_data["consensus_home_implied_prob"]),
                    _prob_to_american(game_data["consensus_away_implied_prob"]),
                    game_data.get("consensus_home_spread"),
                    now)

    # ── Kalshi prediction market probabilities ────────────────────────────────
    try:
        kalshi_markets = get_kalshi_probs(sport_code)
    except Exception as exc:
        log.warning("odds.kalshi_failed", sport=sport_code, error=str(exc))
        kalshi_markets = []

    for market in kalshi_markets:
        game = _match_game(session, sport, market)
        if game is None:
            continue
        prob = market.get("kalshi_implied_prob_home")
        if prob is not None:
            _upsert(session, game.id, "kalshi", "h2h", snapshot,
                    _prob_to_american(prob), _prob_to_american(1.0 - prob), None, now)
            result.rows_inserted += 1

    session.flush()
    log.info("odds.ingest_done", sport=sport_code, snapshot=snapshot,
             inserted=result.rows_inserted, errors=len(result.errors))
    return result


def _upsert(
    session: Session,
    game_id: int,
    bookmaker: str,
    market: str,
    snapshot: str,
    home_price: float | None,
    away_price: float | None,
    home_spread: float | None,
    fetched_at: datetime,
) -> None:
    stmt = pg_insert(GameOdds).values(
        game_id=game_id, bookmaker=bookmaker, market=market, snapshot=snapshot,
        home_price=home_price, away_price=away_price,
        home_spread=home_spread, fetched_at=fetched_at,
    ).on_conflict_do_update(
        index_elements=["game_id", "bookmaker", "market", "snapshot"],
        set_={
            "home_price": home_price, "away_price": away_price,
            "home_spread": home_spread, "fetched_at": fetched_at,
        },
    )
    session.execute(stmt)


def _match_game(session: Session, sport: Any, game_data: dict[str, Any]) -> Game | None:
    """Match an odds game dict to a DB Game by team names + date window."""
    from datetime import timedelta

    commence = game_data.get("commence_time", "")
    if not commence:
        return None

    try:
        game_dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except ValueError:
        return None

    window_start = game_dt - timedelta(hours=12)
    window_end = game_dt + timedelta(hours=12)

    candidates = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= window_start,
            Game.scheduled_utc <= window_end,
        )
        .all()
    )

    home_key = _norm(game_data.get("home_team", ""))
    away_key = _norm(game_data.get("away_team", ""))

    for g in candidates:
        if (
            _norm(g.home_team.name) == home_key
            and _norm(g.away_team.name) == away_key
        ):
            return g
        # Also try abbreviation match
        if (
            _norm(g.home_team.abbrev or "") == home_key
            and _norm(g.away_team.abbrev or "") == away_key
        ):
            return g

    return None


def _norm(name: str) -> str:
    parts = str(name).strip().lower().split()
    return parts[-1] if parts else ""


def _prob_to_american(prob: float) -> float:
    """Convert win probability to American moneyline (no vig)."""
    if prob <= 0 or prob >= 1:
        return -110.0
    if prob >= 0.5:
        return round(-prob / (1.0 - prob) * 100)
    return round((1.0 - prob) / prob * 100)
