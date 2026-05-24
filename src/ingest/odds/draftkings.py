"""DraftKings Sportsbook — odds via their public nash API.

Endpoint: sportsbook-nash.draftkings.com (replaces deprecated sportsbook.draftkings.com/api/odds/v1)
No key required.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1"

_LEAGUE_IDS = {
    "nba": 42648,
    "mlb": 84240,
}

# Game Lines category ID differs by sport
_GAME_LINES_CATEGORY = {
    "nba": 487,
    "mlb": 493,
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
    """Return normalized game-line dicts for all upcoming games in a sport."""
    league_id = _LEAGUE_IDS.get(sport_code.lower())
    if not league_id:
        raise ValueError(f"No DraftKings league ID for sport: {sport_code}")

    category_id = _GAME_LINES_CATEGORY[sport_code.lower()]
    resp = requests.get(
        f"{_BASE}/leagues/{league_id}/categories/{category_id}",
        headers=_HEADERS,
        timeout=15,
    )
    log.info("dk.fetch", sport=sport_code, status=resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    return _parse(data)


def _parse(data: dict[str, Any]) -> list[dict[str, Any]]:
    events = data.get("events", [])
    markets = {m["id"]: m for m in data.get("markets", [])}
    selections = data.get("selections", [])

    # Build event lookup: event_id -> game dict
    event_map: dict[str, dict[str, Any]] = {}
    for ev in events:
        home = away = ""
        for p in ev.get("participants", []):
            role = p.get("venueRole", "")
            # Prefer full rosettaTeamName, fall back to name field
            name = p.get("metadata", {}).get("rosettaTeamName") or p.get("name", "")
            if role == "Home":
                home = name
            elif role == "Away":
                away = name
        if home and away:
            event_map[ev["id"]] = {
                "home_team": home,
                "away_team": away,
                "commence_time": ev.get("startEventDate", ""),
                "home_ml": None,
                "away_ml": None,
                "home_spread": None,
            }

    # Map selections onto events via their market
    for sel in selections:
        market = markets.get(sel.get("marketId", ""))
        if not market:
            continue
        event_id = str(market.get("eventId", ""))
        game = event_map.get(event_id)
        if not game:
            continue

        market_name = (market.get("marketType") or {}).get("name", "").lower()
        outcome_type = sel.get("outcomeType", "")  # "Home" or "Away"
        odds = _parse_american(sel.get("displayOdds", {}).get("american"))
        points = sel.get("points")

        if "moneyline" in market_name:
            if outcome_type == "Home":
                game["home_ml"] = odds
            elif outcome_type == "Away":
                game["away_ml"] = odds
        elif (
            any(k in market_name for k in ("spread", "run line", "puck line"))
            and outcome_type == "Home"
            and points is not None
        ):
            game["home_spread"] = float(points)

    return list(event_map.values())


def _parse_american(value: Any) -> float | None:
    """Parse American odds string, handling Unicode minus sign U+2212 (-)."""
    if value is None:
        return None
    s = str(value).replace("−", "-").replace("+", "").strip()  # noqa: RUF001
    try:
        return float(s)
    except ValueError:
        return None
