"""Redis-backed deduplication for prediction notifications.

Skip re-posting if the same (game_id, target) was sent within the last 6 hours
AND the probability shift is less than 3 percentage points.
Confirmed-lineup updates always re-post (caller passes is_lineup_update=True).
"""
from __future__ import annotations

import redis

from src.core.logging import get_logger

log = get_logger(__name__)

_TTL_SECONDS = 6 * 3600  # 6 hours
_MIN_PROB_SHIFT = 0.03   # 3 pp


def should_send(
    r: redis.Redis,
    game_id: str | int,
    target: str,
    current_prob: float,
    is_lineup_update: bool = False,
) -> bool:
    """Return True if the notification should be sent.

    Always True when is_lineup_update is set.
    Otherwise True only if: first send OR probability shifted >= 3 pp.
    """
    if is_lineup_update:
        _record_send(r, game_id, target, current_prob)
        return True

    key = _key(game_id, target)
    raw = r.get(key)
    if raw is None:
        _record_send(r, game_id, target, current_prob)
        return True

    try:
        last_prob = float(raw)
    except (ValueError, TypeError):
        _record_send(r, game_id, target, current_prob)
        return True

    if abs(current_prob - last_prob) >= _MIN_PROB_SHIFT:
        _record_send(r, game_id, target, current_prob)
        return True

    log.debug(
        "notify.dedup.skip",
        game_id=game_id,
        target=target,
        last_prob=last_prob,
        current_prob=current_prob,
    )
    return False


def _record_send(
    r: redis.Redis,
    game_id: str | int,
    target: str,
    prob: float,
) -> None:
    key = _key(game_id, target)
    r.set(key, str(prob), ex=_TTL_SECONDS)


def _key(game_id: str | int, target: str) -> str:
    return f"notify:dedup:{game_id}:{target}"
