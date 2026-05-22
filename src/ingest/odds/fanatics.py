"""Fanatics Sportsbook — odds fetch via their public web API.

Fanatics Sportsbook (launched 2023, acquired PointsBet US) runs on the
Amelco platform. No API key required — same endpoint their frontend uses.

API base: https://sportsbook.fanatics.com/api/v2
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_BASE = "https://sportsbook.fanatics.com/api/v2"

_SPORT_KEYS = {
    "nba": "basketball/nba",
    "mlb": "baseball/mlb",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.fanatics.com/",
    "x-requested-with": "XMLHttpRequest",
}


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def get_game_lines(sport_code: str) -> list[dict[str, Any]]:
    """Return normalized game-line dicts for upcoming games from Fanatics."""
    sport_key = _SPORT_KEYS.get(sport_code.lower())
    if not sport_key:
        raise ValueError(f"No Fanatics sport key for: {sport_code}")

    resp = requests.get(
        f"{_BASE}/sports/{sport_key}/events",
        params={"marketTypes": "MATCH_ODDS,HANDICAP", "status": "upcoming"},
        headers=_HEADERS,
        timeout=15,
    )
    log.info("fanatics.fetch", sport=sport_code, status=resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    return _parse_events(data)


def _parse_events(data: Any) -> list[dict[str, Any]]:
    """Parse Fanatics event list into normalized game dicts."""
    out: list[dict[str, Any]] = []

    events = data if isinstance(data, list) else data.get("events", data.get("data", []))

    for event in events:
        if not isinstance(event, dict):
            continue

        home_team = _extract_team(event, "home")
        away_team = _extract_team(event, "away")
        if not home_team or not away_team:
            continue

        start_time = (
            event.get("startTime") or event.get("startDate") or event.get("scheduledStart", "")
        )

        game: dict[str, Any] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": start_time,
            "home_ml": None,
            "away_ml": None,
            "home_spread": None,
        }

        markets = event.get("markets", event.get("betOffers", []))
        for market in markets:
            market_type = (
                market.get("type", "")
                or market.get("marketType", "")
                or market.get("betOfferType", {}).get("name", "")
            ).upper()

            outcomes = market.get("outcomes", market.get("selections", []))

            if any(k in market_type for k in ("MATCH_ODDS", "MONEY_LINE", "H2H", "WINNER")):
                for outcome in outcomes:
                    name = (outcome.get("label") or outcome.get("participant") or "").lower()
                    price = _extract_price(outcome)
                    if "home" in name or _is_home(name, home_team):
                        game["home_ml"] = price
                    elif "away" in name or _is_home(name, away_team):
                        game["away_ml"] = price

            if any(k in market_type for k in ("HANDICAP", "SPREAD", "RUN_LINE", "POINT_SPREAD")):
                for outcome in outcomes:
                    name = (outcome.get("label") or outcome.get("participant") or "").lower()
                    line = _safe_float(outcome.get("line") or outcome.get("handicap"))
                    if line is not None and ("home" in name or _is_home(name, home_team)):
                        game["home_spread"] = line

        if game["home_ml"] is not None or game["home_spread"] is not None:
            out.append(game)

    return out


def _extract_team(event: dict[str, Any], side: str) -> str:
    """Pull team name from various possible event shapes."""
    # Shape 1: {homeTeam: {name: ...}, awayTeam: {name: ...}}
    key_map = {"home": ["homeTeam", "home_team", "home"], "away": ["awayTeam", "away_team", "away"]}
    for key in key_map.get(side, []):
        val = event.get(key)
        if isinstance(val, dict):
            return val.get("name") or val.get("teamName") or val.get("shortName") or ""
        if isinstance(val, str) and val:
            return val

    # Shape 2: participants list
    participants = event.get("participants", event.get("teams", []))
    if participants and len(participants) >= 2:
        idx = 0 if side == "home" else 1
        p = participants[idx]
        if isinstance(p, dict):
            return p.get("name") or p.get("teamName") or ""

    return ""


def _extract_price(outcome: dict[str, Any]) -> float | None:
    """Extract American odds from various price formats."""
    # Try American odds directly
    for key in ("oddsAmerican", "americanOdds", "price", "odds"):
        val = outcome.get(key)
        if val is not None:
            s = str(val).replace("+", "")
            try:
                return float(s)
            except ValueError:
                pass

    # Try decimal odds and convert
    for key in ("oddsDecimal", "decimalOdds", "decimalPrice"):
        val = outcome.get(key)
        if val is not None:
            try:
                d = float(val)
                if d >= 2.0:
                    return round((d - 1) * 100)
                elif d > 1.0:
                    return round(-100 / (d - 1))
            except (ValueError, ZeroDivisionError):
                pass

    # Try nested price objects (Kambi/Amelco style)
    price_obj = outcome.get("oddsAmerican") or outcome.get("priceAmerican") or {}
    if isinstance(price_obj, dict):
        val = price_obj.get("american") or price_obj.get("value")
        if val is not None:
            try:
                return float(str(val).replace("+", ""))
            except ValueError:
                pass

    return None


def _is_home(outcome_name: str, team_name: str) -> bool:
    """Fuzzy match: last word of outcome name vs last word of team name."""
    o_parts = outcome_name.strip().lower().split()
    t_parts = team_name.strip().lower().split()
    return bool(o_parts and t_parts and o_parts[-1] == t_parts[-1])


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
