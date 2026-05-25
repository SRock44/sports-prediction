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

# Known individual player prop category IDs per sport (NBA uses category-level endpoints).
# These are curated to include only single-player markets (no combined/duo props).
# DK changes these occasionally; add new ones as discovered.
_PROP_CATEGORIES: dict[str, list[int]] = {
    "nba": [1215, 1216, 1217, 1218, 583, 1293, 1280],  # Pts/Reb/Ast/3PM/PRA/Def milestones
}

# MLB batter/pitcher props live in subcategories — the category-level endpoint only returns
# HR milestones. Each tuple is (category_id, subcategory_id).
_PROP_SUBCATEGORIES: dict[str, list[tuple[int, int]]] = {
    "mlb": [
        (743, 6719),  # Hits O/U → H
        (743, 6607),  # Total Bases O/U → TB
        (743, 8025),  # RBIs O/U → RBI
        (743, 17319),  # Home Runs milestone → HR
        (1031, 15221),  # Strikeouts Thrown O/U → PITCHER_K
        (1031, 17412),  # Earned Runs Allowed O/U → PITCHER_ER
    ],
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
    "strikeouts thrown": "PITCHER_K",
    "pitcher strikeouts": "PITCHER_K",
    "strikeouts": "PITCHER_K",
    "hits allowed": "PITCHER_H",
    "earned runs": "PITCHER_ER",
    "hits": "H",
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
    sport = sport_code.lower()
    league_id = _LEAGUE_IDS.get(sport)
    if not league_id:
        log.warning("dk.props.no_league_id", sport=sport_code)
        return []

    all_props: list[dict[str, Any]] = []

    subcats = _PROP_SUBCATEGORIES.get(sport, [])
    if subcats:
        # Sports with explicit subcategory routing (MLB): each stat lives in its own subcategory.
        # The category-level endpoint only returns milestone HR markets, so we hit subcategories.
        seen_pairs: set[tuple[int, int]] = set()
        for cat_id, subcat_id in subcats:
            if (cat_id, subcat_id) in seen_pairs:
                continue
            seen_pairs.add((cat_id, subcat_id))
            try:
                resp = requests.get(
                    f"{_BASE}/leagues/{league_id}/categories/{cat_id}/subcategories/{subcat_id}",
                    headers=_HEADERS,
                    timeout=15,
                )
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                parsed = _parse_props(data)
                if parsed:
                    log.info(
                        "dk.props.fetched",
                        sport=sport_code,
                        cat=cat_id,
                        subcat=subcat_id,
                        n=len(parsed),
                    )
                    all_props.extend(parsed)
            except Exception as exc:
                log.warning(
                    "dk.props.subcat_failed",
                    sport=sport_code,
                    cat=cat_id,
                    subcat=subcat_id,
                    error=str(exc),
                )
    else:
        # NBA and others: category-level endpoints contain all props directly.
        candidates = list(_PROP_CATEGORIES.get(sport, []))
        if not candidates:
            candidates = _discover_prop_categories(league_id, exclude=[])
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
    """Parse a DK category response into normalized player-prop dicts.

    Handles both traditional Over/Under markets and Milestone markets
    (e.g., "Score 25+" style bets common during NBA/MLB playoffs).
    """
    events = {str(ev["id"]): ev for ev in data.get("events", [])}
    markets = {str(m["id"]): m for m in data.get("markets", [])}
    selections = data.get("selections", [])

    # Group selections by market_id
    ou_sels: dict[str, dict[str, Any]] = {}  # over/under markets
    milestone_sels: dict[str, list[dict[str, Any]]] = {}  # milestone markets

    for sel in selections:
        mid = str(sel.get("marketId", ""))
        outcome = (sel.get("outcomeType") or "").lower()
        if outcome in ("over", "under"):
            if mid not in ou_sels:
                ou_sels[mid] = {}
            ou_sels[mid][outcome] = sel
        elif sel.get("milestoneValue") is not None or (sel.get("label") or "").endswith("+"):
            milestone_sels.setdefault(mid, []).append(sel)

    props: list[dict[str, Any]] = []

    def _extract_event_teams(event_id: str) -> tuple[str, str]:
        event = events.get(str(event_id), {})
        home = away = ""
        for p in event.get("participants", []):
            role = p.get("venueRole", "")
            name = (p.get("metadata") or {}).get("rosettaTeamName") or p.get("name", "")
            if role == "Home":
                home = name
            elif role == "Away":
                away = name
        return home, away

    def _player_from_market(market: dict[str, Any]) -> str:
        for p in market.get("participants", []):
            n = (p.get("metadata") or {}).get("rosterName") or p.get("name", "")
            if n:
                return n
        mname = (market.get("name") or "").strip()
        # "Player Name - Stat Type" format
        parts = mname.split(" - ")
        if len(parts) >= 2:
            return parts[0].strip()
        # Subcategory O/U format: "Luis Arraez Hits O/U" — strip the market type suffix
        mtype = (market.get("marketType") or {}).get("name", "").strip()
        if mtype and mname.endswith(mtype):
            return mname[: -len(mtype)].strip()
        return ""

    # ── Traditional Over/Under markets ───────────────────────────────────────
    for mid, outcomes in ou_sels.items():
        over_sel = outcomes.get("over")
        if not over_sel:
            continue
        market = markets.get(mid)
        if not market:
            continue
        market_name = (market.get("marketType") or {}).get("name", "")
        stat = _detect_stat(market_name)
        if stat is None:
            continue
        player_name = _player_from_market(market)
        if not player_name:
            continue
        line = over_sel.get("points")
        if line is None:
            continue
        under_sel = outcomes.get("under")
        home, away = _extract_event_teams(market.get("eventId", ""))
        event = events.get(str(market.get("eventId", "")), {})
        props.append(
            {
                "player_name": player_name,
                "stat": stat,
                "line": float(line),
                "over_odds": _parse_american(over_sel.get("displayOdds", {}).get("american")),
                "under_odds": _parse_american(under_sel.get("displayOdds", {}).get("american"))
                if under_sel
                else None,
                "home_team": home,
                "away_team": away,
                "commence_time": event.get("startEventDate", ""),
                "bookmaker": "draftkings",
            }
        )

    # ── Milestone markets ("Score 25+") ───────────────────────────────────────
    for mid, sels in milestone_sels.items():
        market = markets.get(mid)
        if not market:
            continue
        market_name = (market.get("marketType") or {}).get("name", "")
        stat = _detect_stat(market_name)
        if stat is None:
            stat = _detect_stat(market.get("name", ""))
        if stat is None:
            continue

        # Player name: check market participants first, then fall back to selection participants
        player_name = _player_from_market(market)
        if not player_name and sels:
            for p in sels[0].get("participants", []):
                if p.get("type") == "Player":
                    n = (p.get("metadata") or {}).get("rosterName") or p.get("name", "")
                    if n:
                        player_name = n
                        break
        if not player_name:
            # Last resort: parse from market display name e.g. "Donovan Mitchell Points"
            mname = market.get("name", "")
            for kw in _MARKET_TO_STAT:
                if kw in mname.lower():
                    player_name = mname.lower().replace(kw, "").strip().title()
                    break

        # Skip combined / multi-player markets
        mtype_lower = (market.get("marketType") or {}).get("name", "").lower()
        mname_lower = (market.get("name") or "").lower()
        if any(
            kw in mtype_lower or kw in mname_lower for kw in ("combined", "either", " & ", " or ")
        ):
            continue
        if not player_name or " & " in player_name or " or " in player_name.lower():
            continue

        # Pick the milestone closest to 50/50 (abs(odds) closest to 100)
        best_sel = min(
            sels,
            key=lambda s: abs(
                abs(_parse_american(s.get("displayOdds", {}).get("american")) or -110) - 100
            ),
        )
        milestone_val = best_sel.get("milestoneValue")
        if milestone_val is None:
            label = (best_sel.get("label") or "").rstrip("+").strip()
            try:
                milestone_val = float(label)
            except ValueError:
                continue

        over_odds = _parse_american(best_sel.get("displayOdds", {}).get("american"))
        event = events.get(str(market.get("eventId", "")), {})
        home, away = _extract_event_teams(market.get("eventId", ""))
        props.append(
            {
                "player_name": player_name,
                "stat": stat,
                "line": float(milestone_val) - 0.5,  # "25+" → line of 24.5
                "over_odds": over_odds,
                "under_odds": None,
                "home_team": home,
                "away_team": away,
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
