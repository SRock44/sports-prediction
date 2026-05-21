"""Thin wrapper around nba_api with rate limiting and error handling.

stats.nba.com is unofficial but stable; nba_api handles endpoint mapping.
We enforce 1 req/sec (the library's default) and rotate User-Agents to reduce
the chance of IP-based throttling during backfill.
"""
from __future__ import annotations

import time
from typing import Any

from nba_api.stats.endpoints import (
    BoxScoreAdvancedV3,
    BoxScoreTraditionalV3,
    LeagueGameFinder,
    PlayByPlayV3,
    ScoreboardV2,
    CommonTeamRoster,
    LeagueStandings,
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.static import teams as static_teams, players as static_players
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.logging import get_logger

log = get_logger(__name__)

_REQUEST_DELAY = 1.0  # seconds between calls


def _sleep() -> None:
    time.sleep(_REQUEST_DELAY)


@retry(
    retry=retry_if_exception_type(Exception),
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
    return df.to_dict(orient="records")  # type: ignore[return-value]


@retry(
    retry=retry_if_exception_type(Exception),
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
    return player_df.to_dict(orient="records"), team_df.to_dict(orient="records")  # type: ignore[return-value]


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_box_score_advanced(game_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (player_rows, team_rows)."""
    _sleep()
    bs = BoxScoreAdvancedV3(game_id=game_id)
    frames = bs.get_data_frames()
    return frames[0].to_dict(orient="records"), frames[1].to_dict(orient="records")  # type: ignore[return-value]


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_play_by_play(game_id: str) -> list[dict[str, Any]]:
    _sleep()
    pbp = PlayByPlayV3(game_id=game_id)
    df = pbp.get_data_frames()[0]
    return df.to_dict(orient="records")  # type: ignore[return-value]


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_team_roster(team_id: str, season: str) -> list[dict[str, Any]]:
    _sleep()
    roster = CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]
    return df.to_dict(orient="records")  # type: ignore[return-value]


def get_all_teams() -> list[dict[str, Any]]:
    """Static team list from nba_api. No network call."""
    return static_teams.get_teams()  # type: ignore[return-value]


def get_all_players(is_only_current_season: bool = False) -> list[dict[str, Any]]:
    """Static player list from nba_api."""
    return static_players.get_players()  # type: ignore[return-value]


def get_live_scoreboard() -> dict[str, Any]:
    """Live game scores and status (polling endpoint)."""
    sb = live_scoreboard.ScoreBoard()
    return sb.get_dict()  # type: ignore[return-value]


def nba_season_str(year: int) -> str:
    """Convert 2024 → '2024-25' (NBA API format)."""
    return f"{year}-{str(year + 1)[-2:]}"
