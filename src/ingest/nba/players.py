"""NBA player roster sync and injury-report ingestion.

The official NBA injury report is published as a PDF daily at:
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_<DATE>_<TIME>.pdf

We parse it with pdfplumber. The PDF format changes occasionally; the parser is
fault-tolerant and falls back to empty on parse failure.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any

import httpx
import pdfplumber
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import utc_now
from src.db.models import Player, Injury, Sport, Team
from src.ingest.common import IngestResult
from src.ingest.nba.client import get_all_teams, get_all_players, get_team_roster, nba_season_str

log = get_logger(__name__)

_INJURY_REPORT_URL_TEMPLATE = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"
)

_STATUS_NORMALISE = {
    "out": "out",
    "questionable": "questionable",
    "probable": "probable",
    "gtd": "questionable",  # game-time decision
    "day-to-day": "questionable",
    "dtd": "questionable",
}


def sync_teams(session: Session) -> IngestResult:
    """Sync static team list from nba_api into the database."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    for t in get_all_teams():
        team_id_ext = str(t["id"])
        existing = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
        if existing is None:
            team = Team(
                sport_id=sport.id,
                external_id=team_id_ext,
                name=t["full_name"],
                abbrev=t["abbreviation"],
                conference=t.get("conference"),
                division=t.get("division"),
                meta={"city": t.get("city"), "state": t.get("state"), "year_founded": t.get("year_founded")},
            )
            session.add(team)
            result.rows_inserted += 1
        else:
            existing.name = t["full_name"]
            existing.abbrev = t["abbreviation"]
            result.rows_updated += 1

    session.flush()
    return result


def sync_players(session: Session, season_year: int) -> IngestResult:
    """Sync player rosters for all teams for a given season."""
    result = IngestResult()
    sport = _get_or_create_sport(session)
    season_str = nba_season_str(season_year)

    teams = session.query(Team).filter_by(sport_id=sport.id).all()
    for team in teams:
        try:
            roster_rows = get_team_roster(team.external_id, season_str)
        except Exception as exc:
            log.warning("nba.players.roster_failed", team=team.abbrev, error=str(exc))
            continue

        for row in roster_rows:
            player_id_ext = str(row.get("PLAYER_ID", ""))
            if not player_id_ext:
                continue

            existing = session.query(Player).filter_by(
                sport_id=sport.id, external_id=player_id_ext
            ).first()
            if existing is None:
                player = Player(
                    sport_id=sport.id,
                    external_id=player_id_ext,
                    full_name=row.get("PLAYER", player_id_ext),
                    primary_position=row.get("POSITION"),
                    meta={"jersey": row.get("NUM"), "height": row.get("HEIGHT"), "weight": row.get("WEIGHT")},
                )
                session.add(player)
                result.rows_inserted += 1
            else:
                existing.primary_position = row.get("POSITION")
                result.rows_updated += 1

    session.flush()
    return result


def ingest_injury_report(session: Session) -> IngestResult:
    """Download and parse today's NBA injury report PDF."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    # Injury report dates are Eastern Time (the NBA's timezone for publishing).
    today = datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    candidate_urls = [
        _INJURY_REPORT_URL_TEMPLATE.format(date=today, time="0500PM"),
        _INJURY_REPORT_URL_TEMPLATE.format(date=today, time="0630PM"),
        _INJURY_REPORT_URL_TEMPLATE.format(date=today, time="0700PM"),
    ]

    pdf_bytes: bytes | None = None
    for url in candidate_urls:
        try:
            resp = httpx.get(url, timeout=20, follow_redirects=True)
            if resp.status_code == 200:
                pdf_bytes = resp.content
                log.info("nba.injury_report.downloaded", url=url, size=len(pdf_bytes))
                break
        except Exception:
            continue

    if pdf_bytes is None:
        log.warning("nba.injury_report.not_found", date=today)
        return result

    injuries = _parse_injury_pdf(pdf_bytes)
    for entry in injuries:
        player = session.query(Player).filter(
            Player.sport_id == sport.id,
            Player.full_name.ilike(f"%{entry['player_name']}%"),
        ).first()
        if player is None:
            log.debug("nba.injury.player_not_found", name=entry["player_name"])
            continue

        injury = Injury(
            player_id=player.id,
            status=entry["status"],
            reason=entry.get("reason"),
            reported_at=utc_now(),
            source="nba_injury_report",
        )
        session.add(injury)
        result.rows_inserted += 1

    session.flush()
    return result


def _parse_injury_pdf(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract player injury entries from NBA injury report PDF.

    The report has columns: Team | Player | Current Status | Reason
    Format changes between seasons; this is a best-effort parser.
    """
    entries: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table is None:
                    continue
                for row in table:
                    if row is None or len(row) < 3:
                        continue
                    # Skip header row
                    if row[0] and row[0].upper() in ("TEAM", "GAME DATE"):
                        continue
                    player_name = (row[1] or "").strip()
                    raw_status = (row[2] or "").strip().lower()
                    reason = (row[3] or "").strip() if len(row) > 3 else None

                    if not player_name:
                        continue

                    status = _STATUS_NORMALISE.get(raw_status, raw_status)
                    entries.append({"player_name": player_name, "status": status, "reason": reason})
    except Exception as exc:
        log.warning("nba.injury.parse_failed", error=str(exc))
    return entries


def _get_or_create_sport(session: Session) -> Sport:
    sport = session.query(Sport).filter_by(code="nba").first()
    if sport is None:
        sport = Sport(code="nba")
        session.add(sport)
        session.flush()
    return sport
