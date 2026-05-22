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
    game: Any,
    as_of: datetime,
) -> dict[str, Any]:
    """Full MLB game feature vector.

    Args:
        game: A Game ORM instance.
        as_of: The cutoff timestamp for feature computation (typically scheduled_utc - 1h).
    """
    import pandas as pd
    game_id: int = game.id
    home_team_id: int = game.home_team_id
    away_team_id: int = game.away_team_id
    sport_id: int = game.sport_id

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
    matchup["run_diff_diff_last5"] = home_feats["run_diff_last5"] - away_feats["run_diff_last5"]
    matchup["woba_diff_last10"] = home_feats["woba_last10"] - away_feats["woba_last10"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]
    matchup["win_pct_diff_last10"] = home_feats["win_pct_last10"] - away_feats["win_pct_last10"]
    matchup["win_pct_season_diff"] = home_feats["win_pct_season"] - away_feats["win_pct_season"]
    matchup["streak_diff"] = home_feats["streak"] - away_feats["streak"]
    matchup["bullpen_rest_diff"] = away_feats["bullpen_ip_last3d"] - home_feats["bullpen_ip_last3d"]

    # ── Starting pitcher features ─────────────────────────────────────────────
    home_sp = _get_confirmed_starter(session, game_id, home_team_id, as_of)
    away_sp = _get_confirmed_starter(session, game_id, away_team_id, as_of)
    matchup.update(_pitcher_features(home_sp, prefix="home_sp"))
    matchup.update(_pitcher_features(away_sp, prefix="away_sp"))
    matchup["sp_xfip_diff"] = matchup.get("home_sp_xfip", 4.0) - matchup.get("away_sp_xfip", 4.0)

    # SP rolling form: last 3 starts ERA, K%, BB%
    home_sp_id = home_sp.get("playerId") or home_sp.get("player_id") if home_sp else None
    away_sp_id = away_sp.get("playerId") or away_sp.get("player_id") if away_sp else None
    matchup.update(_sp_rolling_form(session, home_sp_id, as_of, prefix="home_sp"))
    matchup.update(_sp_rolling_form(session, away_sp_id, as_of, prefix="away_sp"))
    matchup["sp_form_era_diff"] = matchup.get("home_sp_form_era", 4.50) - matchup.get("away_sp_form_era", 4.50)
    matchup["sp_form_k_pct_diff"] = matchup.get("home_sp_form_k_pct", 0.22) - matchup.get("away_sp_form_k_pct", 0.22)

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


def _sp_rolling_form(
    session: Session,
    player_id: int | str | None,
    as_of: datetime,
    prefix: str,
    n_starts: int = 4,
) -> dict[str, Any]:
    """Last N starts ERA, K%, BB% for a starting pitcher."""
    defaults = {
        f"{prefix}_form_era": 4.50,
        f"{prefix}_form_k_pct": 0.22,
        f"{prefix}_form_bb_pct": 0.08,
        f"{prefix}_form_known": 0,
    }
    if player_id is None:
        return defaults
    try:
        result = session.execute(
            text("""
                SELECT pgs.stats->'pitching' AS pit
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE pgs.player_id = :pid
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                  AND pgs.stats->'pitching' IS NOT NULL
                ORDER BY g.scheduled_utc DESC
                LIMIT :n
            """),
            {"pid": player_id, "as_of": as_of, "n": n_starts},
        )
        rows = [r.pit for r in result if r.pit]
        if not rows:
            return defaults

        eras, k_pcts, bb_pcts = [], [], []
        for pit in rows:
            er = pit.get("earnedRuns") or pit.get("earnedRunsAllowed")
            ip = pit.get("inningsPitched")
            k = pit.get("strikeOuts")
            bb = pit.get("baseOnBalls") or pit.get("walks")
            bf = pit.get("battersFaced") or pit.get("pitchesThrown")

            if er is not None and ip and float(ip) > 0:
                eras.append(float(er) * 9.0 / float(ip))
            if k is not None and bf and float(bf) > 0:
                k_pcts.append(float(k) / float(bf))
            if bb is not None and bf and float(bf) > 0:
                bb_pcts.append(float(bb) / float(bf))

        from src.features.common import rolling_mean
        return {
            f"{prefix}_form_era": rolling_mean(eras, n_starts) or 4.50,
            f"{prefix}_form_k_pct": rolling_mean(k_pcts, n_starts) or 0.22,
            f"{prefix}_form_bb_pct": rolling_mean(bb_pcts, n_starts) or 0.08,
            f"{prefix}_form_known": int(bool(eras)),
        }
    except Exception:
        return defaults


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
