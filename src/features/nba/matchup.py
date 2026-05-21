"""NBA matchup-level feature assembly.

Combines home/away team features, Elo difference, H2H history,
and returns a flat feature dict ready for XGBoost.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.features.common import (
    compute_elo_series,
    load_team_game_stats_before,
    haversine_km,
)
from src.features.nba.team import build_team_features

log = get_logger(__name__)


def build_matchup_features(
    session: Session,
    game_id: int,
    home_team_id: int,
    away_team_id: int,
    scheduled_utc: datetime,
    sport_id: int,
) -> dict[str, Any]:
    """Assemble the full matchup feature vector for a single NBA game."""
    from src.core.time import as_of_for_game
    as_of = as_of_for_game(scheduled_utc)

    # ── Elo ratings (computed from all games before as_of) ────────────────────
    elo_ratings = _get_elo_ratings(session, sport_id, as_of)
    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Team-level features ───────────────────────────────────────────────────
    home_feats = build_team_features(
        session, home_team_id, as_of, home_elo, opponent_id=away_team_id, is_home=True
    )
    away_feats = build_team_features(
        session, away_team_id, as_of, away_elo, opponent_id=home_team_id, is_home=False
    )

    # Prefix team features
    matchup: dict[str, Any] = {}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── Cross features ────────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    matchup["elo_diff_with_hca"] = (home_elo + 100.0) - away_elo
    matchup["elo_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(matchup["elo_diff_with_hca"]) / 400.0))

    matchup["net_rtg_diff_last5"] = home_feats["net_rtg_last5"] - away_feats["net_rtg_last5"]
    matchup["net_rtg_diff_last10"] = home_feats["net_rtg_last10"] - away_feats["net_rtg_last10"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]

    # ── Head-to-head (last 5 regular season meetings) ─────────────────────────
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, limit=5)
    matchup["h2h_home_wins"] = h2h["home_wins"]
    matchup["h2h_total"] = h2h["total"]
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)

    # ── Venue travel ──────────────────────────────────────────────────────────
    venue_feats = _get_travel_features(session, game_id, home_team_id, away_team_id, as_of)
    matchup.update(venue_feats)

    return matchup


def _get_elo_ratings(session: Session, sport_id: int, as_of: datetime) -> dict[int, float]:
    """Replay Elo from all completed games before as_of."""
    import pandas as pd
    result = session.execute(
        text("""
            SELECT id, home_team_id, away_team_id, scheduled_utc,
                   CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_won
            FROM games
            WHERE sport_id = :sport_id
              AND scheduled_utc < :as_of
              AND status = 'final'
              AND home_score IS NOT NULL
            ORDER BY scheduled_utc
        """),
        {"sport_id": sport_id, "as_of": as_of},
    )
    rows = [dict(r._mapping) for r in result]
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    return compute_elo_series(df)


def _get_h2h(
    session: Session,
    home_team_id: int,
    away_team_id: int,
    as_of: datetime,
    limit: int = 5,
) -> dict[str, int]:
    result = session.execute(
        text("""
            SELECT home_score, away_score, home_team_id
            FROM games
            WHERE ((home_team_id = :home AND away_team_id = :away)
                OR (home_team_id = :away AND away_team_id = :home))
              AND scheduled_utc < :as_of
              AND status = 'final'
              AND home_score IS NOT NULL
            ORDER BY scheduled_utc DESC
            LIMIT :limit
        """),
        {"home": home_team_id, "away": away_team_id, "as_of": as_of, "limit": limit},
    )
    rows = list(result)
    home_wins = 0
    for row in rows:
        if row.home_team_id == home_team_id and row.home_score > row.away_score:
            home_wins += 1
        elif row.home_team_id == away_team_id and row.away_score > row.home_score:
            home_wins += 1
    return {"home_wins": home_wins, "total": len(rows)}


def _get_travel_features(
    session: Session,
    game_id: int,
    home_team_id: int,
    away_team_id: int,
    as_of: datetime,
) -> dict[str, Any]:
    """Compute travel distance for away team's most recent road trip."""
    # Fetch away team's last game venue
    result = session.execute(
        text("""
            SELECT v.lat, v.lon
            FROM games g
            JOIN venues v ON v.id = g.venue_id
            WHERE (g.home_team_id = :team OR g.away_team_id = :team)
              AND g.scheduled_utc < :as_of
              AND g.status = 'final'
              AND v.lat IS NOT NULL
            ORDER BY g.scheduled_utc DESC
            LIMIT 1
        """),
        {"team": away_team_id, "as_of": as_of},
    )
    prev = result.first()

    current_venue = session.execute(
        text("""
            SELECT v.lat, v.lon
            FROM games g
            JOIN venues v ON v.id = g.venue_id
            WHERE g.id = :game_id AND v.lat IS NOT NULL
        """),
        {"game_id": game_id},
    ).first()

    if prev and current_venue and prev.lat and current_venue.lat:
        away_travel = haversine_km(prev.lat, prev.lon, current_venue.lat, current_venue.lon)
    else:
        away_travel = 0.0

    return {"away_travel_km": away_travel}
