"""MLB player-level feature engineering for batter and pitcher props."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from src.features.common import load_player_game_stats_before, load_injuries_before, rolling_mean


_BATTER_STATS = {
    "H": ("batting", "hits"),
    "HR": ("batting", "homeRuns"),
    "RBI": ("batting", "rbi"),
    "TB": ("batting", "totalBases"),
    "K": ("batting", "strikeOuts"),
    "BB": ("batting", "baseOnBalls"),
}

_PITCHER_STATS = {
    "K": ("pitching", "strikeOuts"),
    "ER": ("pitching", "earnedRuns"),
    "OUTS": ("pitching", "outs"),
    "BB": ("pitching", "baseOnBalls"),
}


def build_batter_features(
    session: Session,
    player_id: int,
    opponent_pitcher_throws: str,
    stadium_factor: float,
    as_of_utc: datetime,
    stat: str = "H",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """~50 features for an MLB batter props model."""
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=25)
    injury_records = load_injuries_before(session, player_id, as_of_utc)

    feats: dict[str, Any] = {}
    feats["injury_status"] = _encode_injury(injury_records)
    feats["season_game_weight"] = min(1.0, games_played_this_season / 20.0)
    feats["opp_pitcher_left"] = int(opponent_pitcher_throws == "L")
    feats["stadium_factor"] = stadium_factor

    if not games:
        _fill_batter_defaults(feats, stat)
        return feats

    stat_vals: list[float] = []
    pa_list: list[float] = []
    vs_same_hand: list[float] = []

    section, col = _BATTER_STATS.get(stat, ("batting", stat.lower()))

    for g in games:
        raw = g["stats"] or {}
        batting = raw.get("batting", raw)
        val = _safe_float(batting.get(col))
        ab = _safe_float(batting.get("atBats"))
        pa = _safe_float(batting.get("plateAppearances") or ab)

        if val is not None:
            stat_vals.append(val)
        if pa is not None:
            pa_list.append(pa)

    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = rolling_mean(stat_vals, w) or 0.0

    feats[f"{stat}_std_last10"] = float(np.std(stat_vals[-10:])) if len(stat_vals) >= 3 else 0.0
    feats["pa_last5"] = rolling_mean(pa_list, 5) or 3.5
    feats["pa_last10"] = rolling_mean(pa_list, 10) or 3.5

    return feats


def build_pitcher_features(
    session: Session,
    player_id: int,
    as_of_utc: datetime,
    stat: str = "K",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Features for MLB pitcher props model."""
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=15)
    injury_records = load_injuries_before(session, player_id, as_of_utc)

    feats: dict[str, Any] = {}
    feats["injury_status"] = _encode_injury(injury_records)
    feats["season_game_weight"] = min(1.0, games_played_this_season / 10.0)

    if not games:
        _fill_pitcher_defaults(feats, stat)
        return feats

    stat_vals: list[float] = []
    outs_list: list[float] = []
    era_list: list[float] = []

    section, col = _PITCHER_STATS.get(stat, ("pitching", stat.lower()))

    for g in games:
        raw = g["stats"] or {}
        pitching = raw.get("pitching", raw)
        val = _safe_float(pitching.get(col))
        outs = _safe_float(pitching.get("outs"))
        er = _safe_float(pitching.get("earnedRuns"))

        if val is not None:
            stat_vals.append(val)
        if outs is not None:
            outs_list.append(outs)
        if er is not None and outs is not None and outs > 0:
            ip = outs / 3.0
            era_list.append(er / ip * 9.0 if ip > 0 else 4.50)

    for w in [3, 5, 10]:
        feats[f"{stat}_last{w}"] = rolling_mean(stat_vals, w) or 0.0

    feats["outs_per_start_last5"] = rolling_mean(outs_list, 5) or 15.0
    feats["era_last5"] = rolling_mean(era_list, 5) or 4.50

    return feats


def _encode_injury(records: list[dict[str, Any]]) -> float:
    if not records:
        return 1.0
    status = records[0].get("status", "active")
    return {"active": 1.0, "il_10": 0.0, "il_60": 0.0, "questionable": 0.7}.get(status, 0.8)


def _fill_batter_defaults(feats: dict[str, Any], stat: str) -> None:
    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = 0.0
    feats[f"{stat}_std_last10"] = 0.0
    feats["pa_last5"] = 3.5
    feats["pa_last10"] = 3.5


def _fill_pitcher_defaults(feats: dict[str, Any], stat: str) -> None:
    for w in [3, 5, 10]:
        feats[f"{stat}_last{w}"] = 0.0
    feats["outs_per_start_last5"] = 15.0
    feats["era_last5"] = 4.50


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
