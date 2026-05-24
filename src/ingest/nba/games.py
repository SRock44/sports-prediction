"""Ingest NBA games, box scores, and play-by-play into Postgres."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import ensure_utc, utc_now
from src.db.models import Game, Player, PlayerGameStats, Sport, Team, TeamGameStats
from src.ingest.common import IngestResult
from src.ingest.nba.client import (
    get_box_score_advanced,
    get_box_score_traditional,
    get_league_game_finder,
    get_scoreboard_for_date,
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

        # LeagueGameFinder returns one row per team per game; group both rows by GAME_ID
        # so we can correctly distinguish home (MATCHUP contains "vs.") from away ("@").
        games_map: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            game_id_ext = str(row["GAME_ID"])
            games_map.setdefault(game_id_ext, []).append(row)

        for game_id_ext, team_rows in games_map.items():
            try:
                _upsert_game_row(session, sport, game_id_ext, team_rows, season_year)
                result.rows_inserted += 1
                result.last_external_id = game_id_ext
            except Exception as exc:
                log.warning("nba.games.upsert_failed", game_id=game_id_ext, error=str(exc))
                result.errors.append(f"{game_id_ext}: {exc}")

    session.flush()
    return result


def _upsert_game_row(
    session: Session,
    sport: Sport,
    game_id_ext: str,
    team_rows: list[dict[str, Any]],
    season_year: int,
) -> None:
    # NBA MATCHUP format: "LAL vs. GSW" (home team row) or "GSW @ LAL" (away team row)
    home_row = next((r for r in team_rows if "vs." in r.get("MATCHUP", "")), None)
    away_row = next((r for r in team_rows if " @ " in r.get("MATCHUP", "")), None)

    if home_row is None or away_row is None:
        log.warning(
            "nba.games.matchup_parse_failed",
            game_id=game_id_ext,
            matchups=[r.get("MATCHUP") for r in team_rows],
        )
        return

    home_team = _resolve_team(session, sport, home_row)
    away_team = _resolve_team(session, sport, away_row)
    if home_team is None or away_team is None:
        return

    game_date_str: str = home_row.get("GAME_DATE", "")
    try:
        scheduled_utc = ensure_utc(
            datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%S")
            if "T" in game_date_str
            else datetime.strptime(game_date_str, "%Y-%m-%d").replace(hour=0, minute=0)
        )
    except ValueError:
        scheduled_utc = datetime.now(UTC)

    now_utc = datetime.now(UTC)
    status = "final" if scheduled_utc <= now_utc else "scheduled"

    existing = session.query(Game).filter_by(sport_id=sport.id, external_id=game_id_ext).first()
    if existing is None:
        game = Game(
            sport_id=sport.id,
            external_id=game_id_ext,
            season=season_year,
            scheduled_utc=scheduled_utc,
            status=status,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_score=home_row.get("PTS") if status == "final" else None,
            away_score=away_row.get("PTS") if status == "final" else None,
            meta={"season_type": home_row.get("SEASON_TYPE", "")},
        )
        session.add(game)
    else:
        if status == "final":
            existing.status = "final"
            existing.home_score = home_row.get("PTS")
            existing.away_score = away_row.get("PTS")


def _resolve_team(session: Session, sport: Sport, row: dict[str, Any]) -> Team | None:
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


_EDT = timezone(timedelta(hours=-4))  # playoffs run in EDT


def _parse_tip_off_utc(game_date: date, status_text: str) -> datetime:
    """Parse '8:00 pm ET' + date into UTC. Falls back to 17:00 UTC (noon ET)."""
    m = re.match(r"(\d+):(\d+)\s*(am|pm)\s*ET", status_text.strip(), re.IGNORECASE)
    if m:
        hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        local = datetime(game_date.year, game_date.month, game_date.day, hour, minute, tzinfo=_EDT)
        return local.astimezone(UTC)
    return datetime(game_date.year, game_date.month, game_date.day, 17, 0, tzinfo=UTC)


def _resolve_team_by_id(session: Session, sport: Sport, team_id: int) -> Team | None:
    """Look up a Team by numeric NBA team ID."""
    return session.query(Team).filter_by(sport_id=sport.id, external_id=str(team_id)).first()


def ingest_upcoming_nba_schedule(session: Session, days_ahead: int = 7) -> IngestResult:
    """Upsert upcoming NBA games for the next N days using ScoreboardV2."""
    from src.core.time import nba_season_for_date

    result = IngestResult()
    sport = _get_or_create_sport(session)
    today = date.today()

    for i in range(days_ahead):
        d = today + timedelta(days=i)
        date_str = d.strftime("%m/%d/%Y")
        try:
            rows = get_scoreboard_for_date(date_str)
        except Exception as exc:
            log.warning("nba.upcoming.fetch_failed", date=date_str, error=str(exc))
            continue

        for row in rows:
            game_id_ext = str(row.get("GAME_ID", ""))
            if not game_id_ext:
                continue

            home_team = _resolve_team_by_id(session, sport, row.get("HOME_TEAM_ID", 0))
            away_team = _resolve_team_by_id(session, sport, row.get("VISITOR_TEAM_ID", 0))
            if home_team is None or away_team is None:
                log.warning(
                    "nba.upcoming.team_not_found",
                    game_id=game_id_ext,
                    home_id=row.get("HOME_TEAM_ID"),
                    away_id=row.get("VISITOR_TEAM_ID"),
                )
                continue

            status_id = int(row.get("GAME_STATUS_ID", 1))
            status_text = str(row.get("GAME_STATUS_TEXT", ""))
            if status_id == 3:
                status = "final"
            elif status_id == 2:
                status = "in_progress"
            else:
                status = "scheduled"

            scheduled_utc = _parse_tip_off_utc(d, status_text)
            season_year = nba_season_for_date(d)

            existing = (
                session.query(Game).filter_by(sport_id=sport.id, external_id=game_id_ext).first()
            )
            if existing is None:
                session.add(
                    Game(
                        sport_id=sport.id,
                        external_id=game_id_ext,
                        season=season_year,
                        scheduled_utc=scheduled_utc,
                        status=status,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                    )
                )
                result.rows_inserted += 1
            else:
                # Always update status; only update time if game hasn't started
                existing.status = status
                if status == "scheduled":
                    existing.scheduled_utc = scheduled_utc
                result.rows_updated += 1

        session.flush()
        log.info("nba.upcoming.date_done", date=date_str, rows=len(rows))

    return result


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
        if team is None:
            log.warning(
                "nba.box_score.player_team_not_found",
                game=game_ext_id,
                player=pid,
                team_ext_id=team_id_ext,
            )
            continue

        existing_pgs = (
            session.query(PlayerGameStats).filter_by(game_id=game.id, player_id=player.id).first()
        )
        if existing_pgs is None:
            pgs = PlayerGameStats(
                game_id=game.id,
                player_id=player.id,
                team_id=team.id,
                recorded_at=utc_now(),
                stats=stats,
            )
            session.add(pgs)
            result.rows_inserted += 1
        else:
            existing_pgs.stats = stats
            result.rows_updated += 1

    # Index advanced team stats by teamId for merge below
    team_adv_map: dict[str, dict[str, Any]] = {str(r.get("teamId", "")): r for r in team_adv}

    # Team stats — merge traditional + advanced
    for row in team_trad:
        team_id_ext = str(row.get("teamId", ""))
        team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
        if team is None:
            continue

        merged_stats: dict[str, Any] = {"traditional": row}
        if team_id_ext in team_adv_map:
            merged_stats["advanced"] = team_adv_map[team_id_ext]

        existing_tgs = (
            session.query(TeamGameStats).filter_by(game_id=game.id, team_id=team.id).first()
        )
        if existing_tgs is None:
            tgs = TeamGameStats(
                game_id=game.id,
                team_id=team.id,
                recorded_at=utc_now(),
                stats=merged_stats,
            )
            session.add(tgs)
        else:
            existing_tgs.stats = merged_stats

    session.flush()
    return result
