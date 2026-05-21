"""MLB matchup feature assembly: combines starter, bullpen, lineup, park, and Elo."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import compute_elo_series, load_team_game_stats_before
from src.features.mlb.team import build_team_features, _PARK_FACTORS


def build_matchup_features(
    session: Session,
    game_id: int,
    home_team_id: int,
    away_team_id: int,
    scheduled_utc: datetime,
    sport_id: int,
) -> dict[str, Any]:
    """Full MLB game feature vector."""
    from src.core.time import as_of_for_game
    import pandas as pd
    as_of = as_of_for_game(scheduled_utc)

    # ── Elo ───────────────────────────────────────────────────────────────────
    result = session.execute(
        text("""
            SELECT id, home_team_id, away_team_id, scheduled_utc,
                   CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_won
            FROM games
            WHERE sport_id = :sid AND scheduled_utc < :as_of AND status='final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc
        """),
        {"sid": sport_id, "as_of": as_of},
    )
    rows = [dict(r._mapping) for r in result]
    if rows:
        df = pd.DataFrame(rows)
        elo_ratings = compute_elo_series(df)
    else:
        elo_ratings = {}

    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Venue/park factor ─────────────────────────────────────────────────────
    venue_row = session.execute(
        text("SELECT v.meta FROM games g JOIN venues v ON v.id=g.venue_id WHERE g.id=:gid"),
        {"gid": game_id},
    ).first()
    park_abbrev = (venue_row.meta or {}).get("team_abbrev") if venue_row else None

    # ── Team features ─────────────────────────────────────────────────────────
    home_feats = build_team_features(session, home_team_id, as_of, home_elo, park_abbrev, is_home=True)
    away_feats = build_team_features(session, away_team_id, as_of, away_elo, park_abbrev, is_home=False)

    matchup: dict[str, Any] = {}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── Cross features ────────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    matchup["elo_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(home_elo + 50.0 - away_elo) / 400.0))
    matchup["run_diff_diff_last10"] = home_feats["run_diff_last10"] - away_feats["run_diff_last10"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]

    # ── Starting pitcher features ─────────────────────────────────────────────
    home_sp = _get_confirmed_starter(session, game_id, home_team_id, as_of)
    away_sp = _get_confirmed_starter(session, game_id, away_team_id, as_of)
    matchup.update(_pitcher_features(home_sp, prefix="home_sp"))
    matchup.update(_pitcher_features(away_sp, prefix="away_sp"))
    matchup["sp_xfip_diff"] = matchup.get("home_sp_xfip", 4.0) - matchup.get("away_sp_xfip", 4.0)

    # ── Head-to-head ──────────────────────────────────────────────────────────
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, 5)
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)
    matchup["h2h_total"] = h2h["total"]

    return matchup


def _get_confirmed_starter(
    session: Session, game_id: int, team_id: int, as_of: datetime
) -> dict[str, Any] | None:
    """Look up confirmed starting pitcher from lineups table."""
    row = session.execute(
        text("""
            SELECT players FROM lineups
            WHERE game_id = :gid AND team_id = :tid AND source = 'official'
            ORDER BY fetched_at DESC LIMIT 1
        """),
        {"gid": game_id, "tid": team_id},
    ).first()
    if row and row.players:
        for p in row.players:
            if p.get("position") in ("SP", "P") and p.get("batting_order") == 0:
                return p
    return None


def _pitcher_features(pitcher: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if pitcher is None:
        return {
            f"{prefix}_xfip": 4.50,
            f"{prefix}_k_bb_pct": 0.10,
            f"{prefix}_handedness": 0,
            f"{prefix}_known": 0,
        }
    return {
        f"{prefix}_xfip": float(pitcher.get("xfip", 4.50)),
        f"{prefix}_k_bb_pct": float(pitcher.get("k_bb_pct", 0.10)),
        f"{prefix}_handedness": int(pitcher.get("throws", "R") == "L"),
        f"{prefix}_known": 1,
    }


def _get_h2h(
    session: Session, home_id: int, away_id: int, as_of: datetime, limit: int
) -> dict[str, int]:
    result = session.execute(
        text("""
            SELECT home_score, away_score, home_team_id
            FROM games
            WHERE ((home_team_id=:h AND away_team_id=:a) OR (home_team_id=:a AND away_team_id=:h))
              AND scheduled_utc < :as_of AND status='final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc DESC LIMIT :limit
        """),
        {"h": home_id, "a": away_id, "as_of": as_of, "limit": limit},
    )
    rows = list(result)
    wins = sum(
        1 for r in rows
        if (r.home_team_id == home_id and r.home_score > r.away_score)
        or (r.home_team_id == away_id and r.away_score > r.home_score)
    )
    return {"home_wins": wins, "total": len(rows)}
