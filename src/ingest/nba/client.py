"""Thin wrapper around nba_api with rate limiting and error handling.

stats.nba.com is unofficial but stable; nba_api handles endpoint mapping.
We enforce 1 req/sec (the library's default) and rotate User-Agents to reduce
the chance of IP-based throttling during backfill.
"""

from __future__ import annotations

import time
from typing import Any

import requests.exceptions
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.endpoints import (
    BoxScoreAdvancedV3,
    BoxScoreTraditionalV3,
    CommonTeamRoster,
    LeagueGameFinder,
    PlayByPlayV3,
)
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_REQUEST_DELAY = 1.0  # seconds between calls


def _sleep() -> None:
    time.sleep(_REQUEST_DELAY)


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_league_game_finder(
    season_nullable: str,
    season_type_nullable: str = "Regular Season",
) -> list[dict[str, Any]]:
    """Fetch all games for a season. Returns list of game dicts."""
    _sleep()
    finder = LeagueGameFinder(
        season_nullable=season_nullable,
        season_type_nullable=season_type_nullable,
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    return df.to_dict(orient="records")  # type: ignore[no-any-return]


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_box_score_traditional(game_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (player_rows, team_rows)."""
    _sleep()
    bs = BoxScoreTraditionalV3(game_id=game_id)
    frames = bs.get_data_frames()
    player_df = frames[0]
    team_df = frames[1]
    return player_df.to_dict(orient="records"), team_df.to_dict(orient="records")


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_box_score_advanced(game_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (player_rows, team_rows)."""
    _sleep()
    bs = BoxScoreAdvancedV3(game_id=game_id)
    frames = bs.get_data_frames()
    return frames[0].to_dict(orient="records"), frames[1].to_dict(orient="records")


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_play_by_play(game_id: str) -> list[dict[str, Any]]:
    _sleep()
    pbp = PlayByPlayV3(game_id=game_id)
    df = pbp.get_data_frames()[0]
    return df.to_dict(orient="records")  # type: ignore[no-any-return]


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_team_roster(team_id: str, season: str) -> list[dict[str, Any]]:
    _sleep()
    roster = CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]
    return df.to_dict(orient="records")  # type: ignore[no-any-return]


def get_all_teams() -> list[dict[str, Any]]:
    """Static team list from nba_api. No network call."""
    return static_teams.get_teams()  # type: ignore[no-any-return]


def get_all_players(is_only_current_season: bool = False) -> list[dict[str, Any]]:
    """Static player list from nba_api."""
    return static_players.get_players(is_only_current_season)  # type: ignore[no-any-return]


_LIVE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "cdn.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def get_live_scoreboard() -> dict[str, Any]:
    """Live game scores and status (polling endpoint)."""
    sb = live_scoreboard.ScoreBoard(headers=_LIVE_HEADERS)
    return sb.get_dict()  # type: ignore[no-any-return]


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_scoreboard_for_date(game_date: str) -> list[dict[str, Any]]:
    """Fetch game headers for a specific date.

    game_date: 'MM/DD/YYYY' format (NBA API convention).
    Returns list of GameHeader row dicts (one per game).
    """
    _sleep()
    from nba_api.stats.endpoints import ScoreboardV2

    sb = ScoreboardV2(game_date=game_date, league_id="00", day_offset=0)
    df = sb.get_data_frames()[0]  # index 0 = GameHeader
    return df.to_dict(orient="records")  # type: ignore[no-any-return]


def nba_season_str(year: int) -> str:
    """Convert 2024 → '2024-25' (NBA API format)."""
    return f"{year}-{str(year + 1)[-2:]}"
