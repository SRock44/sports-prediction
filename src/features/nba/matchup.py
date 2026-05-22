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
    load_game_odds,
    haversine_km,
)
from src.features.nba.team import build_team_features

log = get_logger(__name__)


def build_matchup_features(
    session: Session,
    game: Any,
    as_of: datetime,
) -> dict[str, Any]:
    """Assemble the full matchup feature vector for a single NBA game.

    Args:
        game: A Game ORM instance.
        as_of: The cutoff timestamp for feature computation (typically scheduled_utc - 1h).
    """
    game_id: int = game.id
    home_team_id: int = game.home_team_id
    away_team_id: int = game.away_team_id
    sport_id: int = game.sport_id

    # ── Elo ratings (computed from all games before as_of) ────────────────────
    elo_ratings = _get_elo_ratings(session, sport_id, as_of)
    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Venue longitude (for timezone fatigue) ────────────────────────────────
    venue_lon = _get_venue_lon(session, game_id)

    # ── Team-level features ───────────────────────────────────────────────────
    home_feats = build_team_features(
        session, home_team_id, as_of, home_elo,
        opponent_id=away_team_id, is_home=True, current_venue_lon=venue_lon,
    )
    away_feats = build_team_features(
        session, away_team_id, as_of, away_elo,
        opponent_id=home_team_id, is_home=False, current_venue_lon=venue_lon,
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
    matchup["net_rtg_diff_last20"] = home_feats["net_rtg_last20"] - away_feats["net_rtg_last20"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]

    matchup["pace_diff_last5"] = home_feats["pace_last5"] - away_feats["pace_last5"]
    matchup["pace_diff_last10"] = home_feats["pace_last10"] - away_feats["pace_last10"]
    matchup["ts_pct_diff_last5"] = home_feats["ts_pct_last5"] - away_feats["ts_pct_last5"]
    matchup["ts_pct_diff_last10"] = home_feats["ts_pct_last10"] - away_feats["ts_pct_last10"]
    matchup["tov_rate_diff_last5"] = home_feats["tov_rate_last5"] - away_feats["tov_rate_last5"]
    matchup["oreb_pct_diff_last10"] = home_feats["oreb_pct_last10"] - away_feats["oreb_pct_last10"]

    matchup["win_pct_diff_last5"] = home_feats["win_pct_last5"] - away_feats["win_pct_last5"]
    matchup["win_pct_diff_last10"] = home_feats["win_pct_last10"] - away_feats["win_pct_last10"]
    matchup["win_pct_diff_last20"] = home_feats["win_pct_last20"] - away_feats["win_pct_last20"]
    matchup["win_pct_season_diff"] = home_feats["win_pct_season"] - away_feats["win_pct_season"]

    matchup["streak_diff"] = home_feats["streak"] - away_feats["streak"]
    matchup["b2b_diff"] = home_feats["b2b"] - away_feats["b2b"]
    matchup["schedule_load_diff"] = (
        home_feats["three_in_four"] + home_feats["four_in_six"]
        - away_feats["three_in_four"] - away_feats["four_in_six"]
    )
    matchup["starter_avail_diff"] = home_feats["starter_availability"] - away_feats["starter_availability"]

    # ── Head-to-head (last 5 regular season meetings) ─────────────────────────
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, limit=5)
    matchup["h2h_home_wins"] = h2h["home_wins"]
    matchup["h2h_total"] = h2h["total"]
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)

    # ── Venue travel ──────────────────────────────────────────────────────────
    venue_feats = _get_travel_features(session, game_id, home_team_id, away_team_id, as_of)
    matchup.update(venue_feats)

    # ── Roster quality cross-features ─────────────────────────────────────────
    matchup["roster_star_pts_diff"] = home_feats["roster_star_pts"] - away_feats["roster_star_pts"]
    matchup["roster_star_ts_diff"] = home_feats["roster_star_ts_pct"] - away_feats["roster_star_ts_pct"]
    matchup["roster_depth_diff"] = home_feats["roster_depth_score"] - away_feats["roster_depth_score"]
    # Away team's road fatigue relative to home rest
    matchup["road_streak_away"] = away_feats["road_game_streak"]
    matchup["tz_change_away"] = away_feats["tz_hours_change"]

    # ── Market signals (odds) ─────────────────────────────────────────────────
    odds = load_game_odds(session, game_id)
    if odds:
        matchup.update(odds)
    else:
        # Defaults when no odds data available yet
        matchup["odds_open_implied_home"] = 0.5
        matchup["odds_close_implied_home"] = 0.5
        matchup["odds_line_move"] = 0.0
        matchup["odds_sharp_move"] = 0
        matchup["odds_open_spread"] = 0.0

    # ── Referee tendencies ────────────────────────────────────────────────────
    matchup.update(_get_referee_features(session, game_id))

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


def _get_venue_lon(session: Session, game_id: int) -> float | None:
    """Return longitude of a game's venue."""
    result = session.execute(
        text("SELECT v.lon FROM games g JOIN venues v ON v.id = g.venue_id WHERE g.id = :gid AND v.lon IS NOT NULL"),
        {"gid": game_id},
    ).first()
    return float(result.lon) if result else None


def _get_referee_features(session: Session, game_id: int) -> dict[str, Any]:
    """Aggregate referee crew tendencies for this game.

    Referees are stored in games.meta['officials'] as a list of names.
    We look up each referee's historical stats from past games they worked.
    Gracefully returns defaults if data isn't available yet.
    """
    defaults = {
        "ref_foul_rate": 42.0,   # avg fouls called per game
        "ref_pace_factor": 1.0,  # relative pace vs league avg
        "ref_home_win_pct": 0.5, # how often home team wins with this crew
    }
    try:
        game_meta = session.execute(
            text("SELECT meta FROM games WHERE id = :gid"), {"gid": game_id}
        ).scalar()
        officials = (game_meta or {}).get("officials", [])
        if not officials:
            return defaults

        # Query historical games officiated by any of these referees
        result = session.execute(
            text("""
                SELECT
                    AVG(CASE WHEN (tgs.stats->'traditional'->>'personalFouls')::float IS NOT NULL
                             THEN (tgs.stats->'traditional'->>'personalFouls')::float ELSE NULL END) AS avg_fouls,
                    AVG(CASE WHEN (tgs.stats->'advanced'->>'pace')::float IS NOT NULL
                             THEN (tgs.stats->'advanced'->>'pace')::float ELSE NULL END) AS avg_pace,
                    AVG(CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END) AS home_win_pct,
                    COUNT(DISTINCT g.id) AS n_games
                FROM games g
                JOIN team_game_stats tgs ON tgs.game_id = g.id
                WHERE g.status = 'final'
                  AND g.meta->'officials' ?| :officials
            """),
            {"officials": officials},
        ).first()

        if result and result.n_games and result.n_games >= 5:
            return {
                "ref_foul_rate": float(result.avg_fouls or 42.0),
                "ref_pace_factor": (result.avg_pace or 100.0) / 100.0,
                "ref_home_win_pct": float(result.home_win_pct or 0.5),
            }
    except Exception:
        pass
    return defaults


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
