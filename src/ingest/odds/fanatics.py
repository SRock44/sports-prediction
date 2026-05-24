"""Fanatics Sportsbook — odds fetch.

Fanatics' external API endpoints are currently unreachable from server environments
(connection refused / SSL rejection). This module returns an empty list gracefully
so ingest continues with DraftKings + FanDuel data.

TODO: Re-enable when Fanatics opens a stable public API or we add a proxy.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)


def get_game_lines(sport_code: str) -> list[dict[str, Any]]:
    log.debug("fanatics.skipped", sport=sport_code, reason="api_unreachable")
    return []
