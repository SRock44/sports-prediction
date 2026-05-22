"""MLB data client wrapping MLB-StatsAPI and pybaseball (Baseball Savant).

mlb-statsapi docs: https://github.com/toddrob99/MLB-StatsAPI
Baseball Savant (via pybaseball): https://baseballsavant.mlb.com/statcast_search
"""

from __future__ import annotations

import time
from typing import Any

import pybaseball
import requests.exceptions
import statsapi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_REQUEST_DELAY = 0.3  # seconds between mlb-statsapi calls
pybaseball.cache.enable()  # disk-cache Statcast calls to avoid re-downloading


def _sleep() -> None:
    time.sleep(_REQUEST_DELAY)


# ── Schedule / games ─────────────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_schedule(start_date: str, end_date: str, sport_id: int = 1) -> list[dict[str, Any]]:
    """Fetch scheduled + completed games between two dates (YYYY-MM-DD)."""
    _sleep()
    return statsapi.schedule(  # type: ignore[no-any-return]
        start_date=start_date,
        end_date=end_date,
        sportId=sport_id,
    )


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_boxscore(game_pk: int) -> dict[str, Any]:
    _sleep()
    return statsapi.boxscore_data(game_pk)  # type: ignore[no-any-return]


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_live_feed(game_pk: int) -> dict[str, Any]:
    """GUMBO live feed — current game state including play-by-play."""
    _sleep()
    return statsapi.get("game", {"gamePk": game_pk})  # type: ignore[no-any-return]


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_roster(team_id: int, season: int) -> list[dict[str, Any]]:
    _sleep()
    data = statsapi.get(
        "team_roster",
        {"teamId": team_id, "season": season, "rosterType": "active"},
    )
    return data.get("roster", [])  # type: ignore[no-any-return]


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_transactions(start_date: str, end_date: str) -> list[dict[str, Any]]:
    """IL moves and transactions in date range."""
    _sleep()
    data = statsapi.get(
        "transactions",
        {"startDate": start_date, "endDate": end_date, "sportId": 1},
    )
    return data.get("transactions", [])  # type: ignore[no-any-return]


def get_all_teams(season: int) -> list[dict[str, Any]]:
    data = statsapi.get("teams", {"season": season, "sportId": 1})
    return data.get("teams", [])  # type: ignore[no-any-return]


# ── Statcast (pybaseball) ─────────────────────────────────────────────────────


def get_statcast_range(start_dt: str, end_dt: str) -> Any:
    """Fetch pitch-level Statcast data. Returns a pandas DataFrame.

    pybaseball splits large ranges automatically and caches on disk.
    """
    return pybaseball.statcast(start_dt=start_dt, end_dt=end_dt, parallel=False)


def get_statcast_batter(player_id: int, start_dt: str, end_dt: str) -> Any:
    return pybaseball.statcast_batter(start_dt=start_dt, end_dt=end_dt, player_id=player_id)


def get_statcast_pitcher(player_id: int, start_dt: str, end_dt: str) -> Any:
    return pybaseball.statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=player_id)


def get_batting_stats(season: int) -> Any:
    """FanGraphs batting leaderboard via pybaseball (wOBA, xwOBA, etc.)."""
    return pybaseball.batting_stats(season, qual=10)


def get_pitching_stats(season: int) -> Any:
    """FanGraphs pitching leaderboard (xFIP, SIERA, K-BB%, etc.)."""
    return pybaseball.pitching_stats(season, qual=10)
