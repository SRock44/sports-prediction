"""FanDuel Sportsbook — odds fetch via their public content API.

No key required. Same API their web frontend uses.
"""
from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.logging import get_logger

log = get_logger(__name__)

_BASE = "https://sbapi.fanduel.com/api"

_SPORT_IDS = {
    "nba": "basketball",
    "mlb": "baseball",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.fanduel.com/",
}

# FanDuel uses an API key embedded in their app; this is the stable public-facing one.
_AK = "FhMFpcPWXMeyZxOx"


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def get_game_lines(sport_code: str) -> list[dict[str, Any]]:
    """Return normalized game-line dicts for upcoming games."""
    sport = _SPORT_IDS.get(sport_code.lower())
    if not sport:
        raise ValueError(f"No FanDuel sport key for: {sport_code}")

    resp = requests.get(
        f"{_BASE}/content-managed-page",
        params={
            "page": "CUSTOM",
            "customPageId": sport,
            "betexRegionISOCode": "GB",
            "_ak": _AK,
            "timezone": "America/New_York",
        },
        headers=_HEADERS,
        timeout=15,
    )
    log.info("fd.fetch", sport=sport_code, status=resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    return _parse_events(data, sport_code)


def _parse_events(data: dict[str, Any], sport_code: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # FanDuel response nests events under 'attachments' → 'events'
    attachments = data.get("attachments", {})
    events = attachments.get("events", {})
    markets = attachments.get("markets", {})

    for event_id, event in events.items():
        if not _is_target_sport(event, sport_code):
            continue

        home_team = event.get("homeTeam", {}).get("name", "")
        away_team = event.get("awayTeam", {}).get("name", "")
        start_time = event.get("openDate", "")

        game = {
            "home_team": home_team, "away_team": away_team,
            "commence_time": start_time,
            "home_ml": None, "away_ml": None, "home_spread": None,
        }

        # Look for moneyline/spread in attached markets
        for market_id in event.get("marketIds", []):
            market = markets.get(str(market_id), {})
            market_type = market.get("marketType", "")

            if "MATCH_ODDS" in market_type or "MONEY_LINE" in market_type:
                runners = market.get("runners", [])
                for runner in runners:
                    win_run_line = runner.get("winRunnerOdds", {})
                    price = _american_from_decimal(win_run_line.get("trueOdds", {}).get("decimalOdds", {}).get("decimalOdds"))
                    if runner.get("teamId") == event.get("homeTeamId"):
                        game["home_ml"] = price
                    else:
                        game["away_ml"] = price

            if "HANDICAP" in market_type or "SPREAD" in market_type:
                runners = market.get("runners", [])
                for runner in runners:
                    if runner.get("teamId") == event.get("homeTeamId"):
                        game["home_spread"] = runner.get("handicap")

        if home_team and away_team:
            out.append(game)

    return out


def _is_target_sport(event: dict[str, Any], sport_code: str) -> bool:
    competition = event.get("competitionName", "").lower()
    return sport_code.lower() in competition


def _american_from_decimal(decimal_odds: Any) -> float | None:
    try:
        d = float(decimal_odds)
        if d >= 2.0:
            return round((d - 1) * 100)
        else:
            return round(-100 / (d - 1))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
