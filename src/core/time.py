"""UTC-aware datetime helpers. Everything in this system is stored and compared UTC."""

from __future__ import annotations

from datetime import UTC, date, datetime


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def utc_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def as_of_for_game(scheduled_utc: datetime) -> datetime:
    """The as-of timestamp used for features: 1 hour before tip-off / first pitch.

    This is the critical constant that prevents leakage: we only use data that
    existed at this point in time when building features for this game.
    """
    from datetime import timedelta

    return scheduled_utc - timedelta(hours=1)


def nba_season_for_date(d: date) -> int:
    """Return the NBA season year for a given date.
    NBA seasons start in October; we label by the calendar year in which the season begins.
    This matches the nba_api convention: 2024 → '2024-25'.
    """
    return d.year if d.month >= 10 else d.year - 1


def mlb_season_for_date(d: date) -> int:
    """MLB season is calendar-year based."""
    return d.year
