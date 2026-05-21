"""Ingest MLB games and box scores from MLB-StatsAPI."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import ensure_utc
from src.db.models import Game, Sport, Team, Venue, PlayerGameStats, TeamGameStats, Player
from src.ingest.common import IngestResult
from src.ingest.mlb.client import get_schedule, get_boxscore, get_all_teams

log = get_logger(__name__)


def _get_or_create_sport(session: Session) -> Sport:
    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        sport = Sport(code="mlb")
        session.add(sport)
        session.flush()
    return sport


def sync_teams(session: Session, season: int) -> IngestResult:
    """Upsert all MLB teams for a season."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    teams_data = get_all_teams(season)
    for t in teams_data:
        team_id_ext = str(t["id"])
        venue_data = t.get("venue", {})
        venue_name = venue_data.get("name", "")

        existing = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
        if existing is None:
            team = Team(
                sport_id=sport.id,
                external_id=team_id_ext,
                name=t.get("name", team_id_ext),
                abbrev=t.get("abbreviation", ""),
                conference=t.get("league", {}).get("name"),
                division=t.get("division", {}).get("name"),
                meta={"venue_name": venue_name, "location": t.get("locationName", "")},
            )
            session.add(team)
            result.rows_inserted += 1
        else:
            existing.name = t.get("name", existing.name)
            existing.abbrev = t.get("abbreviation", existing.abbrev)
            result.rows_updated += 1

    session.flush()
    return result


def ingest_season_schedule(session: Session, season: int) -> IngestResult:
    """Ingest the full schedule for an MLB season, chunked by month."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    # MLB season: March-April opener through October (up to today)
    season_start = date(season, 3, 1)
    season_end = min(date(season, 11, 1), date.today())

    chunk_start = season_start
    while chunk_start < season_end:
        chunk_end = min(chunk_start + timedelta(days=30), season_end)
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        try:
            games = get_schedule(start_str, end_str)
        except Exception as exc:
            log.error("mlb.schedule.fetch_failed", start=start_str, error=str(exc))
            result.errors.append(str(exc))
            chunk_start = chunk_end + timedelta(days=1)
            continue

        for g in games:
            try:
                _upsert_game(session, sport, g, season)
                result.rows_inserted += 1
                result.last_external_id = str(g.get("game_pk", ""))
            except Exception as exc:
                log.warning("mlb.games.upsert_failed", game_pk=g.get("game_pk"), error=str(exc))
                result.errors.append(str(exc))

        session.flush()
        chunk_start = chunk_end + timedelta(days=1)

    return result


def _upsert_game(session: Session, sport: Sport, g: dict[str, Any], season: int) -> None:
    game_pk = str(g.get("game_pk", ""))
    if not game_pk:
        return

    home_team = _resolve_team(session, sport, str(g.get("home_id", "")), g.get("home_name", ""))
    away_team = _resolve_team(session, sport, str(g.get("away_id", "")), g.get("away_name", ""))
    if home_team is None or away_team is None:
        return

    game_date_str: str = g.get("game_date", "")
    try:
        scheduled_utc = ensure_utc(datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        try:
            scheduled_utc = ensure_utc(datetime.strptime(game_date_str[:10], "%Y-%m-%d"))
        except ValueError:
            scheduled_utc = datetime.now(timezone.utc)

    raw_status: str = g.get("status", "").lower()
    status = "final" if "final" in raw_status else ("scheduled" if "scheduled" in raw_status else raw_status)

    existing = session.query(Game).filter_by(sport_id=sport.id, external_id=game_pk).first()
    if existing is None:
        game = Game(
            sport_id=sport.id,
            external_id=game_pk,
            season=season,
            scheduled_utc=scheduled_utc,
            status=status,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_score=g.get("home_score"),
            away_score=g.get("away_score"),
            meta={"game_type": g.get("game_type", "R"), "double_header": g.get("double_header", "N")},
        )
        session.add(game)
    else:
        existing.status = status
        if g.get("home_score") is not None:
            existing.home_score = g["home_score"]
        if g.get("away_score") is not None:
            existing.away_score = g["away_score"]


def _resolve_team(session: Session, sport: Sport, team_id_ext: str, name: str) -> Team | None:
    if not team_id_ext or team_id_ext == "0":
        return None
    team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
    if team is None:
        team = Team(
            sport_id=sport.id,
            external_id=team_id_ext,
            name=name,
            abbrev=name[:3].upper(),
            meta={},
        )
        session.add(team)
        session.flush()
    return team


def ingest_box_score(session: Session, game_pk: str) -> IngestResult:
    """Fetch and store full box score for a completed MLB game."""
    result = IngestResult()

    try:
        box = get_boxscore(int(game_pk))
    except Exception as exc:
        log.error("mlb.boxscore.fetch_failed", game_pk=game_pk, error=str(exc))
        result.errors.append(str(exc))
        return result

    sport = _get_or_create_sport(session)
    game = session.query(Game).filter_by(sport_id=sport.id, external_id=game_pk).first()
    if game is None:
        result.errors.append(f"Game {game_pk} not in DB")
        return result

    for side in ("home", "away"):
        side_data = box.get(side, {})
        team_id_ext = str(side_data.get("team", {}).get("id", ""))
        team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()

        team_stats_row = side_data.get("teamStats", {})
        if team and team_stats_row:
            existing = session.query(TeamGameStats).filter_by(game_id=game.id, team_id=team.id).first()
            if existing is None:
                session.add(TeamGameStats(game_id=game.id, team_id=team.id, stats=team_stats_row))
            else:
                existing.stats = team_stats_row

        for pid_str, p_data in side_data.get("players", {}).items():
            person = p_data.get("person", {})
            player_id_ext = str(person.get("id", ""))
            if not player_id_ext:
                continue

            player = session.query(Player).filter_by(sport_id=sport.id, external_id=player_id_ext).first()
            if player is None:
                player = Player(
                    sport_id=sport.id,
                    external_id=player_id_ext,
                    full_name=person.get("fullName", player_id_ext),
                    primary_position=p_data.get("position", {}).get("abbreviation"),
                    bats=p_data.get("batSide", {}).get("code"),
                    throws=p_data.get("pitchHand", {}).get("code"),
                    meta={},
                )
                session.add(player)
                session.flush()

            stats_payload = {
                "batting": p_data.get("stats", {}).get("batting", {}),
                "pitching": p_data.get("stats", {}).get("pitching", {}),
                "fielding": p_data.get("stats", {}).get("fielding", {}),
                "game_status": p_data.get("gameStatus", {}),
            }

            existing_pgs = session.query(PlayerGameStats).filter_by(
                game_id=game.id, player_id=player.id
            ).first()
            if existing_pgs is None:
                session.add(PlayerGameStats(
                    game_id=game.id, player_id=player.id,
                    team_id=team.id if team else game.home_team_id,
                    stats=stats_payload,
                ))
                result.rows_inserted += 1
            else:
                existing_pgs.stats = stats_payload
                result.rows_updated += 1

    session.flush()
    return result
