"""MLB team-level feature engineering."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import load_team_game_stats_before, rolling_mean

_PARK_FACTORS: dict[str, float] = {
    # Run-factor (1.0 = neutral). Source: FanGraphs park factors (approximate)
    "COL": 1.15, "CIN": 1.06, "TEX": 1.05, "HOU": 1.04, "BOS": 1.03,
    "CHC": 1.02, "MIL": 1.01, "ARI": 1.00, "NYY": 0.99, "PHI": 0.98,
    "ATL": 0.97, "LAD": 0.97, "STL": 0.97, "SD": 0.96, "SF": 0.95,
    "SEA": 0.94, "MIN": 0.97, "CWS": 1.00, "DET": 0.98, "CLE": 0.98,
    "TB": 0.98, "NYM": 0.98, "WSH": 0.99, "BAL": 1.00, "MIA": 0.96,
    "KC": 0.97, "OAK": 0.95, "PIT": 0.97, "TOR": 1.00, "LAA": 0.97,
}


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    home_venue_abbrev: str | None = None,
    is_home: bool = True,
) -> dict[str, Any]:
    """Build ~35 MLB team features."""
    games = load_team_game_stats_before(session, team_id, as_of_utc, limit=20)

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating
    feats["is_home"] = int(is_home)
    feats["park_factor"] = _PARK_FACTORS.get(home_venue_abbrev or "", 1.0)

    if not games:
        _fill_defaults(feats)
        return feats

    runs_scored: list[float] = []
    runs_allowed: list[float] = []
    woba: list[float] = []
    k_pct: list[float] = []
    bb_pct: list[float] = []
    won: list[int] = []

    for g in games:
        is_home_game = g["home_team_id"] == team_id
        rs = g["home_score"] if is_home_game else g["away_score"]
        ra = g["away_score"] if is_home_game else g["home_score"]

        if rs is not None:
            runs_scored.append(float(rs))
        if ra is not None:
            runs_allowed.append(float(ra))

        stats = (g["stats"] or {})
        batting = stats.get("batting", {})
        if batting.get("atBats") and batting.get("hits"):
            ab = float(batting["atBats"])
            if ab > 0:
                obp = (float(batting.get("hits", 0)) + float(batting.get("baseOnBalls", 0))) / ab
                woba.append(obp)
            if batting.get("strikeOuts") and ab > 0:
                k_pct.append(float(batting["strikeOuts"]) / ab)
            if batting.get("baseOnBalls") and ab > 0:
                bb_pct.append(float(batting["baseOnBalls"]) / ab)

        if rs is not None and ra is not None:
            won.append(int(rs > ra))

    for w in [5, 10, 15]:
        feats[f"runs_scored_last{w}"] = rolling_mean(runs_scored, w) or 4.5
        feats[f"runs_allowed_last{w}"] = rolling_mean(runs_allowed, w) or 4.5
        feats[f"run_diff_last{w}"] = (feats[f"runs_scored_last{w}"] - feats[f"runs_allowed_last{w}"])
        feats[f"woba_last{w}"] = rolling_mean(woba, w) or 0.320
        feats[f"k_pct_last{w}"] = rolling_mean(k_pct, w) or 0.22
        feats[f"bb_pct_last{w}"] = rolling_mean(bb_pct, w) or 0.08

    feats["win_pct_last10"] = rolling_mean(won, 10) or 0.5

    # ── Rest / schedule density ───────────────────────────────────────────────
    most_recent = sorted([g["scheduled_utc"] for g in games], reverse=True)[0] if games else None
    feats["rest_days"] = (as_of_utc - most_recent).total_seconds() / 86400 if most_recent else 2.0

    return feats


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in [5, 10, 15]:
        feats[f"runs_scored_last{w}"] = 4.5
        feats[f"runs_allowed_last{w}"] = 4.5
        feats[f"run_diff_last{w}"] = 0.0
        feats[f"woba_last{w}"] = 0.320
        feats[f"k_pct_last{w}"] = 0.22
        feats[f"bb_pct_last{w}"] = 0.08
    feats["win_pct_last10"] = 0.5
    feats["rest_days"] = 2.0
