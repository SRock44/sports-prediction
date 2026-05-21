"""NBA team-level feature engineering.

All features use data strictly before `as_of_utc` — enforced by
load_team_game_stats_before() which filters scheduled_utc < as_of.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.features.common import (
    load_team_game_stats_before,
    load_injuries_before,
    rolling_mean,
    haversine_km,
)

log = get_logger(__name__)

_WINDOWS = [5, 10, 20]


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    opponent_id: int | None = None,
    is_home: bool = True,
) -> dict[str, Any]:
    """Build ~60 team features for game-winner model."""
    games = load_team_game_stats_before(session, team_id, as_of_utc, limit=30)

    feats: dict[str, Any] = {}

    # ── Elo ──────────────────────────────────────────────────────────────────
    feats["elo"] = elo_rating
    feats["is_home"] = int(is_home)

    if not games:
        # No history — return priors
        for w in _WINDOWS:
            for stat in ["net_rtg", "off_rtg", "def_rtg", "pace"]:
                feats[f"{stat}_last{w}"] = 0.0
        feats["rest_days"] = 2.0
        feats["b2b"] = 0
        feats["three_in_four"] = 0
        feats["four_in_six"] = 0
        feats["travel_km"] = 0.0
        feats["home_win_pct"] = 0.5
        feats["away_win_pct"] = 0.5
        feats["starter_availability"] = 1.0
        return feats

    # ── Extract time-series of game outcomes and stats ────────────────────────
    game_dates: list[datetime] = []
    off_rtgs: list[float] = []
    def_rtgs: list[float] = []
    net_rtgs: list[float] = []
    paces: list[float] = []
    won: list[int] = []
    home_flags: list[int] = []

    for g in games:
        stats = g["stats"] or {}
        adv = stats.get("advanced", {})
        off_rtg = _safe_float(adv.get("offensiveRating") or adv.get("E_OFF_RATING"))
        def_rtg = _safe_float(adv.get("defensiveRating") or adv.get("E_DEF_RATING"))
        pace = _safe_float(adv.get("pace") or adv.get("PACE"))

        if off_rtg is not None and def_rtg is not None:
            off_rtgs.append(off_rtg)
            def_rtgs.append(def_rtg)
            net_rtgs.append(off_rtg - def_rtg)
        if pace is not None:
            paces.append(pace)

        is_home_game = g["home_team_id"] == team_id
        home_flags.append(int(is_home_game))
        home_score = g["home_score"] or 0
        away_score = g["away_score"] or 0
        if is_home_game:
            won.append(int(home_score > away_score))
        else:
            won.append(int(away_score > home_score))

        game_dates.append(g["scheduled_utc"])

    # ── Rolling efficiency windows ────────────────────────────────────────────
    for w in _WINDOWS:
        feats[f"off_rtg_last{w}"] = rolling_mean(off_rtgs, w) or 110.0
        feats[f"def_rtg_last{w}"] = rolling_mean(def_rtgs, w) or 110.0
        feats[f"net_rtg_last{w}"] = rolling_mean(net_rtgs, w) or 0.0
        feats[f"pace_last{w}"] = rolling_mean(paces, w) or 100.0

    # ── Rest & fatigue flags ──────────────────────────────────────────────────
    most_recent_game = max(game_dates) if game_dates else None
    rest_days = (as_of_utc - most_recent_game).total_seconds() / 86400 if most_recent_game else 3.0
    feats["rest_days"] = min(rest_days, 10.0)  # cap at 10

    game_dates_sorted = sorted(game_dates, reverse=True)
    feats["b2b"] = int(rest_days < 1.5)
    feats["three_in_four"] = int(_games_in_window(game_dates_sorted, as_of_utc, days=4) >= 3)
    feats["four_in_six"] = int(_games_in_window(game_dates_sorted, as_of_utc, days=6) >= 4)

    # ── Home/away splits ──────────────────────────────────────────────────────
    home_games = [w for w, h in zip(won, home_flags) if h == 1]
    away_games_res = [w for w, h in zip(won, home_flags) if h == 0]
    feats["home_win_pct"] = float(np.mean(home_games)) if home_games else 0.5
    feats["away_win_pct"] = float(np.mean(away_games_res)) if away_games_res else 0.5
    feats["overall_win_pct_last10"] = rolling_mean(won, 10) or 0.5

    # ── Travel ───────────────────────────────────────────────────────────────
    # Simplified: we'd look up venue coordinates; default 0 if not available
    feats["travel_km"] = 0.0  # populated by matchup builder if venue coords available

    # ── Starter availability (injury-adjusted) ────────────────────────────────
    # Requires player-level data; placeholder here, overridden by matchup builder
    feats["starter_availability"] = 1.0

    return feats


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _games_in_window(dates_desc: list[datetime], as_of: datetime, days: int) -> int:
    from datetime import timedelta
    cutoff = as_of - timedelta(days=days)
    return sum(1 for d in dates_desc if d >= cutoff)
