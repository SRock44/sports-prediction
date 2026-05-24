"""Fetch and store opening/closing lines from DraftKings, FanDuel, and Kalshi."""

from __future__ import annotations

from datetime import UTC, datetime
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
    now = datetime.now(UTC)

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
        if game is None and sport_code.lower() == "nba":
            game = _create_nba_game_from_odds(session, sport, game_data)
        if game is None:
            continue

        book_map = [
            ("dk", "draftkings"),
            ("fd", "fanduel"),
            ("fan", "fanatics"),
        ]
        for key, bookmaker in book_map:
            if game_data.get(f"{key}_home_ml") is not None:
                _upsert(
                    session,
                    game.id,
                    bookmaker,
                    "h2h",
                    snapshot,
                    game_data[f"{key}_home_ml"],
                    game_data.get(f"{key}_away_ml"),
                    None,
                    now,
                )
                result.rows_inserted += 1
            if game_data.get(f"{key}_home_spread") is not None:
                _upsert(
                    session,
                    game.id,
                    bookmaker,
                    "spreads",
                    snapshot,
                    None,
                    None,
                    game_data[f"{key}_home_spread"],
                    now,
                )
                result.rows_inserted += 1

        # Store consensus h2h (average of all available books)
        if game_data.get("consensus_home_implied_prob", 0.5) != 0.5:
            _upsert(
                session,
                game.id,
                "consensus",
                "h2h",
                snapshot,
                _prob_to_american(game_data["consensus_home_implied_prob"]),
                _prob_to_american(game_data["consensus_away_implied_prob"]),
                game_data.get("consensus_home_spread"),
                now,
            )

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
            _upsert(
                session,
                game.id,
                "kalshi",
                "h2h",
                snapshot,
                _prob_to_american(prob),
                _prob_to_american(1.0 - prob),
                None,
                now,
            )
            result.rows_inserted += 1

    session.flush()
    log.info(
        "odds.ingest_done",
        sport=sport_code,
        snapshot=snapshot,
        inserted=result.rows_inserted,
        errors=len(result.errors),
    )
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
    stmt = (
        pg_insert(GameOdds)
        .values(
            game_id=game_id,
            bookmaker=bookmaker,
            market=market,
            snapshot=snapshot,
            home_price=home_price,
            away_price=away_price,
            home_spread=home_spread,
            fetched_at=fetched_at,
        )
        .on_conflict_do_update(
            index_elements=["game_id", "bookmaker", "market", "snapshot"],
            set_={
                "home_price": home_price,
                "away_price": away_price,
                "home_spread": home_spread,
                "fetched_at": fetched_at,
            },
        )
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
        if _norm(g.home_team.name) == home_key and _norm(g.away_team.name) == away_key:
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


def _find_team_by_name(session: Any, sport_id: int, name: str) -> Any:
    """Look up a Team by the last word of its name (e.g. 'Pacers' matches 'Indiana Pacers')."""
    from src.db.models import Team

    last = _norm(name)
    return (
        session.query(Team).filter(Team.sport_id == sport_id, Team.name.ilike(f"%{last}%")).first()
    )


def _create_nba_game_from_odds(session: Any, sport: Any, game_data: dict[str, Any]) -> Any:
    """Seed a Game record from DK odds when nba.com is unreachable from this server."""
    from src.core.time import nba_season_for_date
    from src.db.models import Game

    home_name = game_data.get("home_team", "")
    away_name = game_data.get("away_team", "")
    commence = game_data.get("commence_time", "")
    if not (home_name and away_name and commence):
        return None

    try:
        game_dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except ValueError:
        return None

    home_team = _find_team_by_name(session, sport.id, home_name)
    away_team = _find_team_by_name(session, sport.id, away_name)
    if home_team is None or away_team is None:
        log.warning("odds.nba_game_create.team_not_found", home=home_name, away=away_name)
        return None

    # Check again by time window + team IDs to avoid duplicates
    from datetime import timedelta

    existing = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.home_team_id == home_team.id,
            Game.away_team_id == away_team.id,
            Game.scheduled_utc >= game_dt - timedelta(hours=12),
            Game.scheduled_utc <= game_dt + timedelta(hours=12),
        )
        .first()
    )
    if existing:
        return existing

    season = nba_season_for_date(game_dt.date())
    game = Game(
        sport_id=sport.id,
        external_id=f"odds_{home_team.abbrev}_{game_dt.strftime('%Y%m%d')}",
        season=season,
        scheduled_utc=game_dt,
        status="scheduled",
        home_team_id=home_team.id,
        away_team_id=away_team.id,
    )
    session.add(game)
    session.flush()
    log.info("odds.nba_game_created", home=home_name, away=away_name, game_id=game.id)
    return game


def _prob_to_american(prob: float) -> float:
    """Convert win probability to American moneyline (no vig)."""
    if prob <= 0 or prob >= 1:
        return -110.0
    if prob >= 0.5:
        return round(-prob / (1.0 - prob) * 100)
    return round((1.0 - prob) / prob * 100)
