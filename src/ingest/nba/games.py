"""Ingest NBA games, box scores, and play-by-play into Postgres."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import nba_season_for_date, ensure_utc
from src.db.models import Game, Sport, Team, Venue, TeamGameStats, PlayerGameStats, Player
from src.ingest.common import IngestResult, Upserter
from src.ingest.nba.client import (
    get_league_game_finder,
    get_box_score_traditional,
    get_box_score_advanced,
    get_play_by_play,
    nba_season_str,
)

log = get_logger(__name__)


def _get_or_create_sport(session: Session) -> Sport:
    sport = session.query(Sport).filter_by(code="nba").first()
    if sport is None:
        sport = Sport(code="nba")
        session.add(sport)
        session.flush()
    return sport


def ingest_season_schedule(session: Session, season_year: int) -> IngestResult:
    """Pull the full schedule for an NBA season (regular + playoffs) and upsert into `games`."""
    result = IngestResult()
    sport = _get_or_create_sport(session)
    season_str = nba_season_str(season_year)

    log.info("nba.ingest_season_schedule", season=season_str)

    for season_type in ("Regular Season", "Playoffs"):
        try:
            rows = get_league_game_finder(season_str, season_type)
        except Exception as exc:
            log.error("nba.games.fetch_failed", season=season_str, type=season_type, error=str(exc))
            result.errors.append(str(exc))
            continue

        # LeagueGameFinder returns one row per team per game; deduplicate by GAME_ID
        seen: set[str] = set()
        for row in rows:
            game_id_ext = str(row["GAME_ID"])
            if game_id_ext in seen:
                continue
            seen.add(game_id_ext)

            try:
                _upsert_game_row(session, sport, row, season_year)
                result.rows_inserted += 1
                result.last_external_id = game_id_ext
            except Exception as exc:
                log.warning("nba.games.upsert_failed", game_id=game_id_ext, error=str(exc))
                result.errors.append(f"{game_id_ext}: {exc}")

    session.flush()
    return result


def _upsert_game_row(session: Session, sport: Sport, row: dict[str, Any], season_year: int) -> None:
    game_id_ext = str(row["GAME_ID"])

    # Resolve teams by external_id
    home_team = _resolve_team(session, sport, row, home=True)
    away_team = _resolve_team(session, sport, row, home=False)
    if home_team is None or away_team is None:
        return

    # Parse game date
    game_date_str: str = row.get("GAME_DATE", "")
    try:
        scheduled_utc = ensure_utc(
            datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%S")
            if "T" in game_date_str
            else datetime.strptime(game_date_str, "%Y-%m-%d").replace(hour=0, minute=0)
        )
    except ValueError:
        scheduled_utc = datetime.now(timezone.utc)

    existing = session.query(Game).filter_by(sport_id=sport.id, external_id=game_id_ext).first()
    if existing is None:
        game = Game(
            sport_id=sport.id,
            external_id=game_id_ext,
            season=season_year,
            scheduled_utc=scheduled_utc,
            status="final",
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            meta={"season_type": row.get("SEASON_TYPE", "")},
        )
        session.add(game)
    else:
        existing.home_score = row.get("PTS") if row.get("MATCHUP", "").find("vs.") != -1 else None
        existing.status = "final"


def _resolve_team(session: Session, sport: Sport, row: dict[str, Any], home: bool) -> Team | None:
    matchup: str = row.get("MATCHUP", "")
    team_abbrev: str = row.get("TEAM_ABBREVIATION", "")
    team_id_ext: str = str(row.get("TEAM_ID", ""))

    team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
    if team is None:
        team = Team(
            sport_id=sport.id,
            external_id=team_id_ext,
            name=row.get("TEAM_NAME", team_abbrev),
            abbrev=team_abbrev,
            meta={},
        )
        session.add(team)
        session.flush()
    return team


def ingest_box_scores(session: Session, game_ext_id: str) -> IngestResult:
    """Fetch and store traditional + advanced box scores for a completed game."""
    result = IngestResult()

    try:
        player_trad, team_trad = get_box_score_traditional(game_ext_id)
        player_adv, team_adv = get_box_score_advanced(game_ext_id)
    except Exception as exc:
        log.error("nba.box_score.fetch_failed", game=game_ext_id, error=str(exc))
        result.errors.append(str(exc))
        return result

    sport = _get_or_create_sport(session)
    game = session.query(Game).filter_by(sport_id=sport.id, external_id=game_ext_id).first()
    if game is None:
        result.errors.append(f"Game {game_ext_id} not found in DB")
        return result

    # Merge trad + adv player stats by PLAYER_ID
    player_stats_map: dict[str, dict[str, Any]] = {}
    for row in player_trad:
        pid = str(row.get("personId", ""))
        player_stats_map[pid] = {"traditional": row}
    for row in player_adv:
        pid = str(row.get("personId", ""))
        if pid in player_stats_map:
            player_stats_map[pid]["advanced"] = row

    for pid, stats in player_stats_map.items():
        player = session.query(Player).filter_by(sport_id=sport.id, external_id=pid).first()
        if player is None:
            trad = stats.get("traditional", {})
            player = Player(
                sport_id=sport.id,
                external_id=pid,
                full_name=trad.get("playerName", pid),
                primary_position=trad.get("position"),
                meta={},
            )
            session.add(player)
            session.flush()

        team_id_ext = str(stats.get("traditional", {}).get("teamId", ""))
        team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()

        existing_pgs = session.query(PlayerGameStats).filter_by(
            game_id=game.id, player_id=player.id
        ).first()
        if existing_pgs is None:
            pgs = PlayerGameStats(
                game_id=game.id,
                player_id=player.id,
                team_id=team.id if team else game.home_team_id,
                stats=stats,
            )
            session.add(pgs)
            result.rows_inserted += 1
        else:
            existing_pgs.stats = stats
            result.rows_updated += 1

    # Team stats
    for row in team_trad:
        team_id_ext = str(row.get("teamId", ""))
        team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
        if team is None:
            continue

        existing_tgs = session.query(TeamGameStats).filter_by(
            game_id=game.id, team_id=team.id
        ).first()
        if existing_tgs is None:
            tgs = TeamGameStats(game_id=game.id, team_id=team.id, stats={"traditional": row})
            session.add(tgs)
        else:
            existing_tgs.stats = {"traditional": row}

    session.flush()
    return result
