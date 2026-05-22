"""Unit tests for UTC time helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from src.core.time import (
    as_of_for_game,
    ensure_utc,
    mlb_season_for_date,
    nba_season_for_date,
    utc_now,
)


class TestUtcNow:
    def test_is_aware(self):
        now = utc_now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)


class TestEnsureUtc:
    def test_naive_gets_utc(self):
        naive = datetime(2024, 11, 1, 12, 0, 0)
        aware = ensure_utc(naive)
        assert aware.tzinfo == UTC

    def test_aware_non_utc_is_converted(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2024, 11, 1, 7, 0, 0, tzinfo=eastern)
        result = ensure_utc(dt)
        assert result.tzinfo == UTC
        assert result.hour == 12  # 7 EST == 12 UTC


class TestAsOf:
    def test_one_hour_before_game(self):
        tip = datetime(2024, 11, 15, 20, 0, 0, tzinfo=UTC)
        as_of = as_of_for_game(tip)
        assert as_of == datetime(2024, 11, 15, 19, 0, 0, tzinfo=UTC)


class TestNbaSeasonForDate:
    @pytest.mark.parametrize(
        "d,expected",
        [
            (date(2024, 10, 22), 2024),  # opening night
            (date(2025, 6, 15), 2024),  # Finals (still 2024-25 season)
            (date(2024, 9, 30), 2023),  # Before Oct 1 → previous season
            (date(2025, 10, 1), 2025),  # New season opens
        ],
    )
    def test_season(self, d, expected):
        assert nba_season_for_date(d) == expected


class TestMlbSeasonForDate:
    def test_always_calendar_year(self):
        assert mlb_season_for_date(date(2025, 4, 1)) == 2025
        assert mlb_season_for_date(date(2025, 10, 31)) == 2025
