"""Ingest MLB games and box scores from MLB-StatsAPI."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import ensure_utc, utc_now
from src.db.models import Game, Player, PlayerGameStats, Sport, Team, TeamGameStats, Venue
from src.ingest.common import IngestResult
from src.ingest.mlb.client import get_all_teams, get_boxscore, get_schedule

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

        abbrev = t.get("abbreviation", "")
        venue_ext_id = str(venue_data.get("id", ""))

        # Upsert the team's home venue so park-factor lookups work in features
        venue_obj: Venue | None = None
        if venue_ext_id:
            venue_obj = (
                session.query(Venue).filter_by(sport_id=sport.id, external_id=venue_ext_id).first()
            )
            if venue_obj is None:
                venue_obj = Venue(
                    sport_id=sport.id,
                    external_id=venue_ext_id,
                    name=venue_name,
                    meta={"team_abbrev": abbrev},
                )
                session.add(venue_obj)
                session.flush()
            else:
                venue_obj.meta = {**venue_obj.meta, "team_abbrev": abbrev}

        existing = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
        if existing is None:
            team = Team(
                sport_id=sport.id,
                external_id=team_id_ext,
                name=t.get("name", team_id_ext),
                abbrev=abbrev,
                conference=t.get("league", {}).get("name"),
                division=t.get("division", {}).get("name"),
                meta={"venue_name": venue_name, "location": t.get("locationName", "")},
            )
            session.add(team)
            result.rows_inserted += 1
        else:
            existing.name = t.get("name", existing.name)
            existing.abbrev = abbrev
            result.rows_updated += 1

    session.flush()
    return result


def ingest_season_schedule(session: Session, season: int) -> IngestResult:
    """Ingest the full schedule for an MLB season, chunked by month."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    # MLB season: March-April opener through October
    season_start = date(season, 3, 1)
    season_end = date(season, 11, 1)

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
                inserted = _upsert_game(session, sport, g, season)
                if inserted:
                    result.rows_inserted += 1
                result.last_external_id = str(g.get("game_id", ""))
            except Exception as exc:
                log.warning("mlb.games.upsert_failed", game_id=g.get("game_id"), error=str(exc))
                result.errors.append(str(exc))

        session.flush()
        chunk_start = chunk_end + timedelta(days=1)

    return result


def _upsert_game(session: Session, sport: Sport, g: dict[str, Any], season: int) -> bool:
    """Returns True if a game was inserted or updated."""
    game_id = str(g.get("game_id", ""))
    if not game_id:
        return False

    home_team = _resolve_team(session, sport, str(g.get("home_id", "")), g.get("home_name", ""))
    away_team = _resolve_team(session, sport, str(g.get("away_id", "")), g.get("away_name", ""))
    if home_team is None or away_team is None:
        return False

    # Prefer game_datetime (full ISO) over game_date (date-only)
    game_date_str: str = g.get("game_datetime") or g.get("game_date", "")
    try:
        scheduled_utc = ensure_utc(datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        try:
            scheduled_utc = ensure_utc(datetime.strptime(game_date_str[:10], "%Y-%m-%d"))
        except ValueError:
            scheduled_utc = datetime.now(UTC)

    raw_status: str = g.get("status", "").lower()
    status = (
        "final"
        if "final" in raw_status
        else ("scheduled" if "scheduled" in raw_status else raw_status)
    )

    venue_ext_id = str(g.get("venue_id", ""))
    venue_obj = (
        session.query(Venue).filter_by(sport_id=sport.id, external_id=venue_ext_id).first()
        if venue_ext_id
        else None
    )

    stmt = (
        pg_insert(Game.__table__)  # type: ignore[arg-type]
        .values(
            sport_id=sport.id,
            external_id=game_id,
            season=season,
            scheduled_utc=scheduled_utc,
            status=status,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            venue_id=venue_obj.id if venue_obj else None,
            home_score=g.get("home_score"),
            away_score=g.get("away_score"),
            meta={
                "game_type": g.get("game_type", "R"),
                "double_header": g.get("doubleheader", "N"),
            },
        )
        .on_conflict_do_update(
            index_elements=["sport_id", "external_id"],
            set_=dict(
                status=status,
                home_score=g.get("home_score"),
                away_score=g.get("away_score"),
            ),
        )
    )
    session.execute(stmt)
    return True


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

    # Purge existing stats so re-ingestion is idempotent (same fix as NBA).
    session.query(PlayerGameStats).filter_by(game_id=game.id).delete()
    session.query(TeamGameStats).filter_by(game_id=game.id).delete()
    session.flush()

    for side in ("home", "away"):
        side_data = box.get(side, {})
        team_id_ext = str(side_data.get("team", {}).get("id", ""))
        team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()

        team_stats_row = side_data.get("teamStats", {})
        if team and team_stats_row:
            session.add(
                TeamGameStats(
                    game_id=game.id,
                    team_id=team.id,
                    recorded_at=utc_now(),
                    stats=team_stats_row,
                )
            )

        if team is None:
            log.warning(
                "mlb.box_score.team_not_found", game_pk=game_pk, side=side, team_ext_id=team_id_ext
            )

        for _pid_str, p_data in side_data.get("players", {}).items():
            person = p_data.get("person", {})
            player_id_ext = str(person.get("id", ""))
            if not player_id_ext:
                continue

            player = (
                session.query(Player)
                .filter_by(sport_id=sport.id, external_id=player_id_ext)
                .first()
            )
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

            if team is None:
                log.warning(
                    "mlb.box_score.player_team_missing",
                    game_pk=game_pk,
                    player=player_id_ext,
                    side=side,
                )
                continue

            stats_payload = {
                "batting": p_data.get("stats", {}).get("batting", {}),
                "pitching": p_data.get("stats", {}).get("pitching", {}),
                "fielding": p_data.get("stats", {}).get("fielding", {}),
                "game_status": p_data.get("gameStatus", {}),
            }

            session.add(
                PlayerGameStats(
                    game_id=game.id,
                    player_id=player.id,
                    team_id=team.id,
                    recorded_at=utc_now(),
                    stats=stats_payload,
                )
            )
            result.rows_inserted += 1

    session.flush()
    return result
