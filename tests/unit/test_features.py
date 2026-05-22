"""Unit tests for feature math: Elo, rolling windows, prob bar, dedup."""

from __future__ import annotations

import math

import pytest

from src.features.common import (
    elo_expected,
    elo_update,
    exponential_decay_weight,
    haversine_km,
    rolling_mean,
)
from src.notify.discord import _prob_bar

# ── Elo ───────────────────────────────────────────────────────────────────────


class TestElo:
    def test_expected_equal_ratings(self):
        assert elo_expected(1500, 1500) == pytest.approx(0.5, abs=1e-6)

    def test_expected_higher_wins(self):
        assert elo_expected(1600, 1400) > 0.5

    def test_update_winner_gains(self):
        a_new, b_new = elo_update(1500, 1500, score_a=1.0)
        assert a_new > 1500
        assert b_new < 1500

    def test_update_zero_sum(self):
        a_new, b_new = elo_update(1500, 1600, score_a=0.0)
        delta = (a_new - 1500) + (b_new - 1600)
        assert abs(delta) < 1e-9

    def test_update_draws_are_symmetric_at_equal_ratings(self):
        a_new, b_new = elo_update(1500, 1500, score_a=0.5)
        assert a_new == pytest.approx(1500, abs=1e-6)
        assert b_new == pytest.approx(1500, abs=1e-6)

    def test_k_factor_respected(self):
        a_new, _ = elo_update(1500, 1500, score_a=1.0, k=20)
        assert a_new == pytest.approx(1510.0, abs=1e-3)


# ── Rolling mean ──────────────────────────────────────────────────────────────


class TestRollingMean:
    def test_simple(self):
        assert rolling_mean([1.0, 2.0, 3.0], window=3) == pytest.approx(2.0)

    def test_window_shorter_than_series(self):
        assert rolling_mean([10.0, 20.0, 30.0], window=2) == pytest.approx(25.0)

    def test_empty_returns_nan(self):
        assert math.isnan(rolling_mean([], window=5))

    def test_min_periods_returns_nan_when_not_met(self):
        assert math.isnan(rolling_mean([1.0], window=5, min_periods=2))

    def test_min_periods_met(self):
        assert rolling_mean([1.0, 2.0], window=5, min_periods=2) == pytest.approx(1.5)


# ── Exponential decay weights ─────────────────────────────────────────────────


class TestDecayWeight:
    def test_zero_days_is_one(self):
        assert exponential_decay_weight(0, lam=0.3) == pytest.approx(1.0)

    def test_higher_lam_decays_faster(self):
        slow = exponential_decay_weight(90, lam=0.1)
        fast = exponential_decay_weight(90, lam=0.5)
        assert fast < slow

    def test_one_year_lam_half(self):
        # exp(-0.5 * 365/365) = exp(-0.5) ≈ 0.6065
        w = exponential_decay_weight(365, lam=0.5)
        assert w == pytest.approx(math.exp(-0.5), rel=1e-4)


# ── Haversine ─────────────────────────────────────────────────────────────────


class TestHaversine:
    def test_same_point_zero(self):
        assert haversine_km(40.7, -74.0, 40.7, -74.0) == pytest.approx(0.0, abs=1e-3)

    def test_nyc_to_la_approx(self):
        km = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
        # ~3940 km
        assert 3800 < km < 4100

    def test_symmetry(self):
        a = haversine_km(51.5, -0.1, 48.9, 2.3)
        b = haversine_km(48.9, 2.3, 51.5, -0.1)
        assert a == pytest.approx(b, rel=1e-6)


# ── Discord prob bar ──────────────────────────────────────────────────────────


class TestProbBar:
    def test_fifty_fifty(self):
        bar = _prob_bar(0.5, width=10)
        assert bar.count("█") == 5
        assert bar.count("░") == 5
        assert "50%" in bar

    def test_full(self):
        bar = _prob_bar(1.0, width=10)
        assert bar.count("█") == 10
        assert bar.count("░") == 0

    def test_zero(self):
        bar = _prob_bar(0.0, width=10)
        assert bar.count("█") == 0
        assert bar.count("░") == 10
