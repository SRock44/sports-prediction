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

# Player prop category IDs per sport.  DK uses a "Player Props" top-level
# category; subcategory IDs cover individual stat markets.
_PROP_CATEGORY = {
    "nba": 1000,
    "mlb": 1001,
}

# Stat keyword → canonical stat name (matches train_props.py _props_stats).
# Checked against the lowercase market-type name.
_MARKET_TO_STAT: dict[str, str] = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "3-point": "3PM",
    "three-point": "3PM",
    "threes": "3PM",
    "pra": "PRA",
    "pts+reb": "PRA",
    "strikeouts": "K",
    "pitcher strikeouts": "PITCHER_K",
    "hits allowed": "PITCHER_H",
    "earned runs": "PITCHER_ER",
    " hits": "H",
    "home runs": "HR",
    "total bases": "TB",
    "rbi": "RBI",
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


def get_player_props(sport_code: str) -> list[dict[str, Any]]:
    """Return normalized player-prop dicts from DraftKings for today's games.

    Each dict has:
      player_name, stat, line, over_odds, under_odds,
      home_team, away_team, commence_time, bookmaker='draftkings'

    Returns an empty list if the category is unavailable or no props are found.
    """
    league_id = _LEAGUE_IDS.get(sport_code.lower())
    if not league_id:
        log.warning("dk.props.no_league_id", sport=sport_code)
        return []

    # First try the known prop category; fall back to category discovery.
    known_cat = _PROP_CATEGORY.get(sport_code.lower())
    candidates = [known_cat] if known_cat else []
    candidates += _discover_prop_categories(league_id, exclude=candidates)

    all_props: list[dict[str, Any]] = []
    seen_cats: set[int] = set()

    for cat_id in candidates:
        if cat_id in seen_cats:
            continue
        seen_cats.add(cat_id)
        try:
            resp = requests.get(
                f"{_BASE}/leagues/{league_id}/categories/{cat_id}",
                headers=_HEADERS,
                timeout=15,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            parsed = _parse_props(data)
            if parsed:
                log.info("dk.props.fetched", sport=sport_code, cat=cat_id, n=len(parsed))
                all_props.extend(parsed)
        except Exception as exc:
            log.warning("dk.props.cat_failed", sport=sport_code, cat=cat_id, error=str(exc))

    # Deduplicate by (player_name, stat, line) keeping first occurrence
    seen: set[tuple[str, str, float]] = set()
    deduped: list[dict[str, Any]] = []
    for p in all_props:
        key = (p["player_name"], p["stat"], p["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


def _discover_prop_categories(league_id: int, exclude: list[int]) -> list[int]:
    """Hit the league endpoint and return category IDs whose name contains 'prop'."""
    try:
        resp = requests.get(
            f"{_BASE}/leagues/{league_id}",
            headers=_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        cats: list[int] = []
        for cat in data.get("categories", []):
            cid = cat.get("id")
            cname = (cat.get("name") or "").lower()
            if cid and cid not in exclude and ("prop" in cname or "player" in cname):
                cats.append(cid)
        return cats
    except Exception:
        return []


def _detect_stat(market_name: str) -> str | None:
    """Map a DK market-type name to a canonical stat code."""
    name = market_name.lower()
    # Longest-key-first matching avoids "hits" matching "pitcher strikeouts"
    for keyword in sorted(_MARKET_TO_STAT, key=len, reverse=True):
        if keyword in name:
            return _MARKET_TO_STAT[keyword]
    return None


def _parse_props(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a DK category response into normalized player-prop dicts."""
    events = {str(ev["id"]): ev for ev in data.get("events", [])}
    markets = {str(m["id"]): m for m in data.get("markets", [])}

    # Group selections by market_id so we can pair Over/Under
    market_selections: dict[str, dict[str, Any]] = {}
    for sel in data.get("selections", []):
        mid = str(sel.get("marketId", ""))
        outcome = (sel.get("outcomeType") or "").lower()
        if outcome not in ("over", "under"):
            continue
        if mid not in market_selections:
            market_selections[mid] = {}
        market_selections[mid][outcome] = sel

    props: list[dict[str, Any]] = []

    for mid, outcomes in market_selections.items():
        over_sel = outcomes.get("over")
        under_sel = outcomes.get("under")
        if not over_sel:
            continue

        market = markets.get(mid)
        if not market:
            continue

        market_name = (market.get("marketType") or {}).get("name", "")
        stat = _detect_stat(market_name)
        if stat is None:
            continue

        # Player name: prefer explicit participant, fall back to parsing market name
        player_name = ""
        for p in market.get("participants", []):
            n = (p.get("metadata") or {}).get("rosterName") or p.get("name", "")
            if n:
                player_name = n
                break
        if not player_name:
            # "LeBron James - Points O/U 24.5" → "LeBron James"
            parts = market_name.split(" - ")
            if len(parts) >= 2:
                player_name = parts[0].strip()

        if not player_name:
            continue

        line = over_sel.get("points")
        if line is None:
            continue

        over_odds = _parse_american(over_sel.get("displayOdds", {}).get("american"))
        under_odds = (
            _parse_american(under_sel.get("displayOdds", {}).get("american")) if under_sel else None
        )

        event_id = str(market.get("eventId", ""))
        event = events.get(event_id, {})
        home_team = away_team = ""
        for p in event.get("participants", []):
            role = p.get("venueRole", "")
            name = (p.get("metadata") or {}).get("rosettaTeamName") or p.get("name", "")
            if role == "Home":
                home_team = name
            elif role == "Away":
                away_team = name

        props.append(
            {
                "player_name": player_name,
                "stat": stat,
                "line": float(line),
                "over_odds": over_odds,
                "under_odds": under_odds,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event.get("startEventDate", ""),
                "bookmaker": "draftkings",
            }
        )

    return props


def _parse_american(value: Any) -> float | None:
    """Parse American odds string, handling Unicode minus sign U+2212 (-)."""
    if value is None:
        return None
    s = str(value).replace("−", "-").replace("+", "").strip()  # noqa: RUF001
    try:
        return float(s)
    except ValueError:
        return None
