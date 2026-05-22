"""DraftKings Sportsbook — odds fetch via their public web API.

This is the same API their frontend uses. No key required.
Endpoints and league IDs are stable but not officially documented.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_BASE = "https://sportsbook.draftkings.com/api/odds/v1"

# DraftKings internal sport/category IDs
_LEAGUE_IDS = {
    "nba": 42648,
    "mlb": 84240,
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.draftkings.com/",
}


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def get_game_lines(sport_code: str) -> list[dict[str, Any]]:
    """Return list of parsed game-line dicts for all upcoming games in a sport.

    Each dict has: home_team, away_team, commence_time, home_ml, away_ml,
    home_spread, away_spread.
    """
    league_id = _LEAGUE_IDS.get(sport_code.lower())
    if not league_id:
        raise ValueError(f"No DraftKings league ID for sport: {sport_code}")

    resp = requests.get(
        f"{_BASE}/leagues/{league_id}",
        headers=_HEADERS,
        timeout=15,
    )
    log.info("dk.fetch", sport=sport_code, status=resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    return _parse_events(data)


def _parse_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse DraftKings event groups response into normalized game dicts."""
    out: list[dict[str, Any]] = []

    # Find game lines category (moneyline + spread)
    offers_map: dict[str, dict[str, Any]] = {}

    # DK nests events under eventGroups → offerSubcategory → offers
    for event_group in data.get("eventGroup", {}).get("events", []):
        event_id = str(event_group.get("eventId", ""))
        home = event_group.get("teamName1", "")  # home is teamName1 in DK convention
        away = event_group.get("teamName2", "")
        start_time = event_group.get("startDate", "")
        offers_map[event_id] = {
            "home_team": home,
            "away_team": away,
            "commence_time": start_time,
            "home_ml": None,
            "away_ml": None,
            "home_spread": None,
        }

    # Also try the newer API structure
    for ev in _iter_events_v2(data):
        out.append(ev)

    return out if out else list(offers_map.values())


def _iter_events_v2(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle DraftKings' newer nested response shape."""
    out = []
    for cat in data.get("eventGroup", {}).get("offerCategories", []):
        for subcat in cat.get("offerSubcategoryDescriptors", []):
            for sub in subcat.get("offerSubcategory", {}).get("offers", []):
                for offer_group in sub:
                    if not isinstance(offer_group, list):
                        offer_group = [offer_group]
                    for offer in offer_group:
                        try:
                            parsed = _parse_offer(offer)
                            if parsed:
                                out.append(parsed)
                        except Exception:
                            continue
    return out


def _parse_offer(offer: dict[str, Any]) -> dict[str, Any] | None:
    label = (offer.get("label") or "").lower()
    if "game" not in label and "moneyline" not in label and "spread" not in label:
        return None

    outcomes = offer.get("outcomes", [])
    if len(outcomes) < 2:
        return None

    home_outcome = next((o for o in outcomes if o.get("label") == "Home"), None)
    away_outcome = next((o for o in outcomes if o.get("label") == "Away"), None)
    if home_outcome is None:
        home_outcome = outcomes[0]
        away_outcome = outcomes[1]

    return {
        "home_team": home_outcome.get("participant", ""),
        "away_team": away_outcome.get("participant", ""),
        "commence_time": offer.get("startDate", ""),
        "home_ml": _safe_price(home_outcome.get("oddsAmerican")),
        "away_ml": _safe_price(away_outcome.get("oddsAmerican")),
        "home_spread": _safe_float(home_outcome.get("line")),
    }


def _safe_price(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
