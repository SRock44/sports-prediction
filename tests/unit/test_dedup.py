"""Unit tests for notification deduplication logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.notify.dedup import _key, should_send


def _mock_redis(stored_value: str | None = None) -> MagicMock:
    r = MagicMock()
    r.get.return_value = stored_value
    return r


class TestShouldSend:
    def test_first_send_always_true(self):
        r = _mock_redis(None)
        assert should_send(r, game_id=1, target="home_win", current_prob=0.6)

    def test_no_shift_returns_false(self):
        r = _mock_redis("0.6")
        assert not should_send(r, game_id=1, target="home_win", current_prob=0.61)

    def test_shift_above_threshold_returns_true(self):
        r = _mock_redis("0.6")
        assert should_send(r, game_id=1, target="home_win", current_prob=0.64)

    def test_lineup_update_always_true(self):
        r = _mock_redis("0.6")
        assert should_send(r, game_id=1, target="home_win", current_prob=0.6, is_lineup_update=True)

    def test_corrupt_stored_value_treated_as_first_send(self):
        r = _mock_redis("not-a-float")
        assert should_send(r, game_id=1, target="home_win", current_prob=0.5)

    def test_key_format(self):
        assert _key(42, "home_win") == "notify:dedup:42:home_win"
