"""Show only player props where the model has meaningful edge over DK's implied probability.

"Winners" = props where model P(over) - DK implied P(over) >= MIN_EDGE (default 4%).
Sorted best edge first.

Usage:
  python -m scripts.props_winners              # NBA
  python -m scripts.props_winners --sport mlb
  python -m scripts.props_winners --stat PTS
  python -m scripts.props_winners --min-edge 0.05
"""

from __future__ import annotations

import argparse
import difflib
from datetime import UTC, datetime
from typing import Any

from src.db.session import sync_session_factory
from src.ingest.odds.draftkings import get_player_props

_DEFAULT_MIN_EDGE = 0.04


def _american_to_implied(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _fmt_odds(v: float | None) -> str:
    if v is None:
        return "  n/a"
    return f"{int(v):+5d}"


def _load_prop_model(sport: str, stat: str) -> Any | None:
    try:
        from src.db.models import ModelRecord
        from src.db.models import Sport as SportModel
        from src.models.registry import load_model

        with sync_session_factory() as s:
            sport_obj = s.query(SportModel).filter_by(code=sport).first()
            if sport_obj is None:
                return None
            rec = (
                s.query(ModelRecord)
                .filter_by(sport_id=sport_obj.id, kind="props", target=stat, active=True)
                .first()
            )
            if rec is None:
                return None
            return load_model(rec.mlflow_run_id, framework="sklearn")
    except Exception:
        return None


def _load_feat_names(sport: str, stat: str) -> list[str]:
    try:
        import json

        from src.db.models import ModelRecord
        from src.db.models import Sport as SportModel
        from src.models.score import load_model_feature_names

        with sync_session_factory() as s:
            sport_obj = s.query(SportModel).filter_by(code=sport).first()
            if sport_obj is None:
                return []
            rec = (
                s.query(ModelRecord)
                .filter_by(sport_id=sport_obj.id, kind="props", target=stat, active=True)
                .first()
            )
            if rec is None:
                return []
            return json.loads(load_model_feature_names(rec.mlflow_run_id))
    except Exception:
        return []


def _lookup_player(session: Any, sport: str, name: str) -> Any | None:
    from src.db.models import Player
    from src.db.models import Sport as SportModel

    sport_obj = session.query(SportModel).filter_by(code=sport).first()
    if sport_obj is None:
        return None
    players = session.query(Player).filter_by(sport_id=sport_obj.id).all()
    if not players:
        return None
    names = [p.full_name for p in players]
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if not matches:
        return None
    return next(p for p in players if p.full_name == matches[0])


def _get_player_team(session: Any, player_id: int, sport: str) -> Any | None:
    from sqlalchemy import text

    row = session.execute(
        text("""
        SELECT pgs.team_id FROM player_game_stats pgs
        JOIN games g ON g.id = pgs.game_id
        JOIN sports sp ON sp.id = g.sport_id
        WHERE pgs.player_id = :pid AND sp.code = :sport
        ORDER BY g.scheduled_utc DESC LIMIT 1
        """),
        {"pid": player_id, "sport": sport},
    ).fetchone()
    if row is None:
        return None
    from src.db.models import Team

    return session.get(Team, row.team_id)


def _get_opponent(session: Any, team_id: int, sport: str, hours: int) -> Any | None:
    from sqlalchemy import text

    row = session.execute(
        text("""
        SELECT CASE WHEN g.home_team_id = :tid THEN g.away_team_id
                    ELSE g.home_team_id END AS opp_id
        FROM games g JOIN sports sp ON sp.id = g.sport_id
        WHERE sp.code = :sport
          AND (g.home_team_id = :tid OR g.away_team_id = :tid)
          AND g.scheduled_utc BETWEEN NOW() - INTERVAL '4 hours'
                                  AND NOW() + :hrs * INTERVAL '1 hour'
          AND g.status IN ('scheduled','pre-game','in progress','delayed','delayed start')
        ORDER BY g.scheduled_utc LIMIT 1
        """),
        {"tid": team_id, "sport": sport, "hrs": hours},
    ).fetchone()
    if row is None:
        return None
    from src.db.models import Team

    return session.get(Team, row.opp_id)


_MLB_PITCHER_STATS = {"PITCHER_K", "PITCHER_ER", "PITCHER_H"}


def _build_feats(session: Any, player: Any, team: Any, opp: Any, stat: str, sport: str) -> dict:
    try:
        as_of = datetime.now(UTC)
        if sport == "nba":
            from src.features.nba.player import build_player_features

            return build_player_features(
                session=session,
                player_id=player.id,
                team_id=team.id,
                opponent_team_id=opp.id,
                as_of_utc=as_of,
                stat=stat,
            )
        elif sport == "mlb":
            if stat in _MLB_PITCHER_STATS:
                from src.features.mlb.player import build_pitcher_features

                return build_pitcher_features(
                    session=session,
                    player_id=player.id,
                    as_of_utc=as_of,
                    stat=stat.replace("PITCHER_", ""),
                )
            else:
                from src.features.mlb.player import build_batter_features

                return build_batter_features(
                    session=session,
                    player_id=player.id,
                    opponent_pitcher_throws="R",
                    stadium_factor=1.0,
                    as_of_utc=as_of,
                    stat=stat,
                )
    except Exception:
        pass
    return {}


def main(
    sport: str = "nba",
    stat_filter: str | None = None,
    min_edge: float = _DEFAULT_MIN_EDGE,
    hours: int = 36,
) -> None:
    print(f"\nFetching DK {sport.upper()} player props…")
    props = get_player_props(sport)

    if not props:
        print("  No DK prop lines available.")
        return

    if stat_filter:
        props = [p for p in props if p["stat"].upper() == stat_filter.upper()]

    # Pre-load models
    stats_needed = {p["stat"] for p in props}
    models: dict[str, Any] = {}
    feat_names: dict[str, list] = {}
    for stat in stats_needed:
        m = _load_prop_model(sport, stat)
        if m:
            models[stat] = m
            feat_names[stat] = _load_feat_names(sport, stat)

    if not models:
        print(
            "  No active prop models found. "
            "Run 'python -m src.cli train --sport nba --kind props' to train them.\n"
        )
        return

    edges: list[dict[str, Any]] = []

    with sync_session_factory() as session:
        for prop in props:
            stat = prop["stat"]
            model = models.get(stat)
            if model is None:
                continue

            fn = feat_names.get(stat, [])
            if not fn:
                continue

            player = _lookup_player(session, sport, prop["player_name"])
            if player is None:
                continue

            team = _get_player_team(session, player.id, sport)
            if team is None:
                continue

            opp = _get_opponent(session, team.id, sport, hours)
            if opp is None:
                continue

            feats = _build_feats(session, player, team, opp, stat, sport)
            if not feats:
                continue

            import numpy as np

            x = np.array([[feats.get(n, 0.0) for n in fn]], dtype=np.float32)

            try:
                model_over = model.implied_over_probability(x[0], prop["line"])
                model_under = 1.0 - model_over
            except Exception:
                continue

            if prop["over_odds"] is None:
                continue

            dk_over_implied = _american_to_implied(prop["over_odds"])
            dk_under_implied = (
                _american_to_implied(prop["under_odds"]) if prop["under_odds"] else None
            )

            over_edge = model_over - dk_over_implied
            under_edge = (model_under - dk_under_implied) if dk_under_implied else None

            # Get model median for display
            try:
                preds = model.predict(x)
                model_median = float(preds[0.50][0])
            except Exception:
                model_median = None

            if over_edge >= min_edge:
                edges.append(
                    {
                        "player_name": prop["player_name"],
                        "stat": stat,
                        "line": prop["line"],
                        "direction": "OVER",
                        "edge": over_edge,
                        "model_prob": model_over,
                        "dk_implied": dk_over_implied,
                        "odds": prop["over_odds"],
                        "model_median": model_median,
                        "game": f"{prop['away_team']} @ {prop['home_team']}",
                    }
                )
            if under_edge is not None and under_edge >= min_edge:
                edges.append(
                    {
                        "player_name": prop["player_name"],
                        "stat": stat,
                        "line": prop["line"],
                        "direction": "UNDER",
                        "edge": under_edge,
                        "model_prob": model_under,
                        "dk_implied": dk_under_implied,
                        "odds": prop["under_odds"],
                        "model_median": model_median,
                        "game": f"{prop['away_team']} @ {prop['home_team']}",
                    }
                )

    if not edges:
        print(f"  No prop edges ≥ {min_edge:.0%} found.\n")
        return

    edges.sort(key=lambda x: x["edge"], reverse=True)

    now = datetime.now(UTC)
    print(f"\n{'─' * 90}")
    print(
        f"  {sport.upper()} PROP WINNERS — edge ≥ {min_edge:.0%}  "
        f"({now.strftime('%Y-%m-%d %H:%M UTC')})"
    )
    print(f"{'─' * 90}")
    header = (
        f"  {'PLAYER':<24}  {'STAT':<8}  {'LINE':>5}  {'DIR':>5}"
        f"  {'MODEL%':>7}  {'DK%':>7}  {'EDGE':>6}  {'ODDS':>6}  {'MED':>5}"
    )
    print(header)
    print(f"{'─' * 90}")

    for e in edges:
        med_str = f"{e['model_median']:5.1f}" if e["model_median"] is not None else "  n/a"
        print(
            f"  {e['player_name']:<24}  {e['stat']:<8}  {e['line']:>5.1f}"
            f"  {e['direction']:>5}  {e['model_prob']:>6.1%}  {e['dk_implied']:>6.1%}"
            f"  {e['edge']:>+6.1%}  {_fmt_odds(e['odds']):>6}  {med_str}"
        )
        print(f"    → {e['game']}")

    print(f"{'─' * 90}")
    print(f"  {len(edges)} qualifying edge(s)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--stat", default=None)
    parser.add_argument("--min-edge", type=float, default=_DEFAULT_MIN_EDGE)
    parser.add_argument("--hours", type=int, default=36)
    args = parser.parse_args()
    main(args.sport, args.stat, args.min_edge, args.hours)
