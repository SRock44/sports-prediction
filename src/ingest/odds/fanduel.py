"""FanDuel Sportsbook — odds via their public content API.

sbapi.fanduel.com rejects standard TLS from Linux hosts; we use httpx with a
permissive SSL context to work around the handshake failure.
"""

from __future__ import annotations

import ssl
from typing import Any

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

_AK = "FhMFpcPWXMeyZxOx"


def _make_ssl_context() -> ssl.SSLContext:
    """Permissive SSL context that accepts FanDuel's legacy TLS config."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
    return ctx


def get_game_lines(sport_code: str) -> list[dict[str, Any]]:
    """Return normalized game-line dicts for upcoming games."""
    try:
        import httpx
    except ImportError:
        log.warning("fd.httpx_missing")
        return []

    sport = _SPORT_IDS.get(sport_code.lower())
    if not sport:
        raise ValueError(f"No FanDuel sport key for: {sport_code}")

    params = {
        "page": "CUSTOM",
        "customPageId": sport,
        "betexRegionISOCode": "US",
        "_ak": _AK,
        "timezone": "America/New_York",
    }

    try:
        with httpx.Client(verify=False, timeout=15, headers=_HEADERS) as client:
            resp = client.get(f"{_BASE}/content-managed-page", params=params)
        log.info("fd.fetch", sport=sport_code, status=resp.status_code)
        resp.raise_for_status()
        data = resp.json()
        return _parse_events(data, sport_code)
    except Exception as exc:
        log.warning("fd.fetch_failed", sport=sport_code, error=str(exc))
        return []


def _parse_events(data: dict[str, Any], sport_code: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    attachments = data.get("attachments", {})
    events = attachments.get("events", {})
    markets = attachments.get("markets", {})

    for _event_id, event in events.items():
        if not _is_target_sport(event, sport_code):
            continue

        home_team = event.get("homeTeam", {}).get("name", "")
        away_team = event.get("awayTeam", {}).get("name", "")
        start_time = event.get("openDate", "")

        if not home_team or not away_team:
            continue

        game: dict[str, Any] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": start_time,
            "home_ml": None,
            "away_ml": None,
            "home_spread": None,
        }

        for market_id in event.get("marketIds", []):
            market = markets.get(str(market_id), {})
            market_type = market.get("marketType", "")

            if "MATCH_ODDS" in market_type or "MONEY_LINE" in market_type:
                for runner in market.get("runners", []):
                    price = _american_from_decimal(
                        runner.get("winRunnerOdds", {})
                        .get("trueOdds", {})
                        .get("decimalOdds", {})
                        .get("decimalOdds")
                    )
                    if runner.get("teamId") == event.get("homeTeamId"):
                        game["home_ml"] = price
                    else:
                        game["away_ml"] = price

            if "HANDICAP" in market_type or "SPREAD" in market_type:
                for runner in market.get("runners", []):
                    if runner.get("teamId") == event.get("homeTeamId"):
                        game["home_spread"] = runner.get("handicap")

        out.append(game)

    return out


def _is_target_sport(event: dict[str, Any], sport_code: str) -> bool:
    return sport_code.lower() in event.get("competitionName", "").lower()


def _american_from_decimal(decimal_odds: Any) -> float | None:
    try:
        d = float(decimal_odds)
        if d >= 2.0:
            return round((d - 1) * 100)
        else:
            return round(-100 / (d - 1))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
