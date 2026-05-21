"""MLB IL transactions and roster sync."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import utc_now
from src.db.models import Injury, Player, Sport, Team
from src.ingest.common import IngestResult
from src.ingest.mlb.client import get_transactions, get_roster

log = get_logger(__name__)

_IL_TRANSACTION_TYPES = frozenset(["IL Placement", "IL Transfer", "IL Reinstatement", "10 Day IL", "60 Day IL"])


def ingest_il_transactions(session: Session, lookback_days: int = 3) -> IngestResult:
    """Pull IL moves from the last N days and update injury table."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    today = date.today()
    start = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    try:
        transactions = get_transactions(start, end)
    except Exception as exc:
        log.error("mlb.il.fetch_failed", error=str(exc))
        result.errors.append(str(exc))
        return result

    for txn in transactions:
        txn_type: str = txn.get("typeDesc", "")
        if txn_type not in _IL_TRANSACTION_TYPES:
            continue

        person = txn.get("person", {})
        player_id_ext = str(person.get("id", ""))
        if not player_id_ext:
            continue

        player = session.query(Player).filter_by(sport_id=sport.id, external_id=player_id_ext).first()
        if player is None:
            player = Player(
                sport_id=sport.id,
                external_id=player_id_ext,
                full_name=person.get("fullName", player_id_ext),
                meta={},
            )
            session.add(player)
            session.flush()

        if "Reinstatement" in txn_type:
            status = "active"
        elif "Transfer" in txn_type:
            status = "il_60" if "60" in txn_type else "il_10"
        else:
            status = "il_10" if "10" in txn_type else "il_60"

        injury = Injury(
            player_id=player.id,
            status=status,
            reason=txn.get("description"),
            reported_at=utc_now(),
            source="mlb_transactions",
        )
        session.add(injury)
        result.rows_inserted += 1

    session.flush()
    log.info("mlb.il.ingested", count=result.rows_inserted)
    return result


def sync_roster(session: Session, team_id_ext: str, season: int) -> IngestResult:
    """Sync active roster for one team."""
    result = IngestResult()
    sport = _get_or_create_sport(session)

    team = session.query(Team).filter_by(sport_id=sport.id, external_id=team_id_ext).first()
    if team is None:
        return result

    try:
        roster = get_roster(int(team_id_ext), season)
    except Exception as exc:
        log.warning("mlb.roster.fetch_failed", team=team_id_ext, error=str(exc))
        result.errors.append(str(exc))
        return result

    for entry in roster:
        person = entry.get("person", {})
        player_id_ext = str(person.get("id", ""))
        if not player_id_ext:
            continue

        existing = session.query(Player).filter_by(sport_id=sport.id, external_id=player_id_ext).first()
        pos_data = entry.get("position", {})
        if existing is None:
            player = Player(
                sport_id=sport.id,
                external_id=player_id_ext,
                full_name=person.get("fullName", player_id_ext),
                primary_position=pos_data.get("abbreviation"),
                meta={"jersey_number": entry.get("jerseyNumber"), "status": entry.get("status", {}).get("code")},
            )
            session.add(player)
            result.rows_inserted += 1
        else:
            existing.primary_position = pos_data.get("abbreviation", existing.primary_position)
            result.rows_updated += 1

    session.flush()
    return result


def _get_or_create_sport(session: Session) -> Sport:
    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        sport = Sport(code="mlb")
        session.add(sport)
        session.flush()
    return sport
