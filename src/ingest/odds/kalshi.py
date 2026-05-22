"""Kalshi prediction market client — documented public API.

Kalshi is a CFTC-regulated prediction market. Their API is fully documented
at https://trading-api.kalshi.com/trade-api/v2/openapi.json

No API key required for read-only market data. The markets are YES/NO contracts
on event outcomes, priced 0-100 cents (= 0-100% probability).

These complement traditional moneyline odds nicely — Kalshi reflects
sophisticated retail + some institutional money, so their prices are
meaningful probability signals even when lines haven't moved at books.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# Kalshi series tickers for sports
_SERIES = {
    "nba": "NBAWIN",  # NBA game winner markets
    "mlb": "MLBWIN",  # MLB game winner markets
}

_HEADERS = {
    "User-Agent": "prediction-app/1.0",
    "Accept": "application/json",
}


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def get_game_markets(sport_code: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return normalized game-line dicts from Kalshi prediction markets.

    Kalshi prices (yes_bid + yes_ask) / 2 ≈ implied win probability directly.
    No conversion from American odds needed.
    """
    series = _SERIES.get(sport_code.lower())
    if not series:
        raise ValueError(f"No Kalshi series for sport: {sport_code}")

    resp = requests.get(
        f"{_BASE}/markets",
        params={  # type: ignore[arg-type]
            "series_ticker": series,
            "status": "open",
            "limit": limit,
        },
        headers=_HEADERS,
        timeout=15,
    )
    log.info("kalshi.fetch", sport=sport_code, status=resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    return _parse_markets(data.get("markets", []))


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def get_market_by_ticker(ticker: str) -> dict[str, Any] | None:
    """Fetch a single market by its ticker (e.g. 'NBAWIN-LAL-BOS-20260104')."""
    resp = requests.get(
        f"{_BASE}/markets/{ticker}",
        headers=_HEADERS,
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("market")  # type: ignore[no-any-return]


def _parse_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Kalshi market dicts to our normalized format."""
    out = []
    for m in markets:
        title = m.get("title", "")
        close_time = m.get("close_time", "")

        # Kalshi YES price = probability home team wins (in cents, 0-100)
        yes_bid = m.get("yes_bid", 50)
        yes_ask = m.get("yes_ask", 50)
        implied_prob = ((yes_bid + yes_ask) / 2.0) / 100.0  # normalize to 0-1

        # Parse team names from title (format: "Team A vs Team B")
        teams = _parse_teams_from_title(title)
        if not teams:
            continue

        out.append(
            {
                "home_team": teams[0],
                "away_team": teams[1],
                "commence_time": close_time,
                "kalshi_implied_prob_home": implied_prob,
                "kalshi_yes_bid": yes_bid / 100.0,
                "kalshi_yes_ask": yes_ask / 100.0,
                "kalshi_ticker": m.get("ticker", ""),
            }
        )

    return out


def _parse_teams_from_title(title: str) -> list[str] | None:
    """Extract [home, away] from titles like 'LAL to beat BOS' or 'LAL vs BOS'."""
    title_lower = title.lower()
    for sep in [" vs ", " @ ", " to beat "]:
        if sep in title_lower:
            parts = title_lower.split(sep, 1)
            return [p.strip().upper() for p in parts]
    return None
