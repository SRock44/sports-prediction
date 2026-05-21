"""UTC-aware datetime helpers. Everything in this system is stored and compared UTC."""
from __future__ import annotations

from datetime import datetime, timezone, date
from typing import overload


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_of_for_game(scheduled_utc: datetime) -> datetime:
    """The as-of timestamp used for features: 1 hour before tip-off / first pitch.

    This is the critical constant that prevents leakage: we only use data that
    existed at this point in time when building features for this game.
    """
    from datetime import timedelta
    return scheduled_utc - timedelta(hours=1)


def nba_season_for_date(d: date) -> int:
    """Return the NBA season year for a given date.
    NBA seasons start in October; we label by the year the Finals are played.
    """
    return d.year if d.month >= 10 else d.year - 1


def mlb_season_for_date(d: date) -> int:
    """MLB season is calendar-year based."""
    return d.year
