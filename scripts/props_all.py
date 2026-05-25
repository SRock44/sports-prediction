"""Show all player prop model predictions vs DraftKings lines for today's games.

For each DK prop line, loads the active prop model (if trained) and shows
the model's quantile distribution + implied P(over) vs the DK implied probability.

Usage:
  python -m scripts.props_all              # NBA, all stats
  python -m scripts.props_all --sport mlb
  python -m scripts.props_all --stat PTS   # one stat only
  python -m scripts.props_all --hours 36
"""

from __future__ import annotations

import argparse
import difflib
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from src.db.session import sync_session_factory
from src.ingest.odds.draftkings import get_player_props


# ── Odds math ─────────────────────────────────────────────────────────────────


def _american_to_implied(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _fmt_odds(v: float | None) -> str:
    if v is None:
        return "  n/a"
    return f"{int(v):+5d}"


# ── Model helpers ─────────────────────────────────────────────────────────────


def _load_prop_model(sport: str, stat: str) -> Any | None:
    """Load active prop model for a sport+stat from MLflow. Returns bundle or None."""
    try:
        from src.db.models import ModelRecord, Sport as SportModel
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
            run_id = rec.mlflow_run_id

        return load_model(run_id, framework="sklearn")
    except Exception:
        return None


def _load_prop_feature_names(sport: str, stat: str) -> list[str]:
    try:
        import json

        from src.db.models import ModelRecord, Sport as SportModel
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
    """Fuzzy-match a player by name within a sport."""
    from src.db.models import Player, Sport as SportModel

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
    """Return the team a player most recently played for."""
    from sqlalchemy import text

    row = session.execute(
        text("""
        SELECT pgs.team_id
        FROM player_game_stats pgs
        JOIN games g ON g.id = pgs.game_id
        JOIN sports sp ON sp.id = g.sport_id
        WHERE pgs.player_id = :pid AND sp.code = :sport
        ORDER BY g.scheduled_utc DESC
        LIMIT 1
        """),
        {"pid": player_id, "sport": sport},
    ).fetchone()
    if row is None:
        return None

    from src.db.models import Team

    return session.get(Team, row.team_id)


def _get_opponent(session: Any, team_id: int, sport: str, hours: int) -> Any | None:
    """Return the upcoming opponent team for the given team."""
    from sqlalchemy import text

    row = session.execute(
        text("""
        SELECT
            CASE WHEN g.home_team_id = :tid THEN g.away_team_id
                 ELSE g.home_team_id END AS opp_id
        FROM games g
        JOIN sports sp ON sp.id = g.sport_id
        WHERE sp.code = :sport
          AND (g.home_team_id = :tid OR g.away_team_id = :tid)
          AND g.scheduled_utc BETWEEN NOW() - INTERVAL '4 hours'
                                  AND NOW() + :hrs * INTERVAL '1 hour'
          AND g.status IN ('scheduled','pre-game','in progress','delayed','delayed start')
        ORDER BY g.scheduled_utc
        LIMIT 1
        """),
        {"tid": team_id, "sport": sport, "hrs": hours},
    ).fetchone()
    if row is None:
        return None

    from src.db.models import Team

    return session.get(Team, row.opp_id)


_MLB_PITCHER_STATS = {"PITCHER_K", "PITCHER_ER", "PITCHER_H"}


def _build_player_features_safe(
    session: Any, player: Any, team: Any, opponent: Any, stat: str, sport: str
) -> dict[str, Any]:
    try:
        as_of = datetime.now(UTC)
        if sport == "nba":
            from src.features.nba.player import build_player_features

            return build_player_features(
                session=session,
                player_id=player.id,
                team_id=team.id,
                opponent_team_id=opponent.id,
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
                    opponent_pitcher_throws="R",  # default; ~70% of MLB pitchers
                    stadium_factor=1.0,
                    as_of_utc=as_of,
                    stat=stat,
                )
    except Exception:
        pass
    return {}


def _implied_over_prob(model: Any, features: dict, feature_names: list, line: float) -> float | None:
    import numpy as np

    try:
        x = np.array([[features.get(n, 0.0) for n in feature_names]], dtype=np.float32)
        return model.implied_over_probability(x[0], line)
    except Exception:
        return None


def _get_median(model: Any, features: dict, feature_names: list) -> float | None:
    import numpy as np

    try:
        x = np.array([[features.get(n, 0.0) for n in feature_names]], dtype=np.float32)
        preds = model.predict(x)
        return float(preds[0.50][0])
    except Exception:
        return None


# ── Display ────────────────────────────────────────────────────────────────────


def _print_prop_row(
    player_name: str,
    stat: str,
    line: float,
    over_odds: float | None,
    under_odds: float | None,
    model_median: float | None,
    model_over_prob: float | None,
    dk_over_implied: float | None,
    edge: float | None,
) -> None:
    over_str = _fmt_odds(over_odds)
    under_str = _fmt_odds(under_odds)
    med_str = f"{model_median:5.1f}" if model_median is not None else "  n/a"
    prob_str = f"{model_over_prob:.1%}" if model_over_prob is not None else "  n/a "
    imp_str = f"{dk_over_implied:.1%}" if dk_over_implied is not None else "  n/a "
    edge_str = f"{edge:+.1%}" if edge is not None else "   n/a"

    print(
        f"    {player_name:<24}  {stat:<8}  O/U {line:5.1f}"
        f"  model {med_str} ({prob_str} over)"
        f"  DK {over_str}/{under_str}  implied {imp_str}  edge {edge_str}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(sport: str = "nba", stat_filter: str | None = None, hours: int = 36) -> None:
    print(f"\nFetching DK {sport.upper()} player props…")
    props = get_player_props(sport)

    if not props:
        print("  No DK prop lines available.")
        return

    if stat_filter:
        props = [p for p in props if p["stat"].upper() == stat_filter.upper()]
        if not props:
            print(f"  No {stat_filter.upper()} lines from DK.")
            return

    # Pre-load models per stat so we don't reload on every player
    stats_needed = {p["stat"] for p in props}
    models: dict[str, Any] = {}
    feat_names: dict[str, list[str]] = {}
    for stat in stats_needed:
        m = _load_prop_model(sport, stat)
        if m is not None:
            models[stat] = m
            feat_names[stat] = _load_prop_feature_names(sport, stat)

    if not models:
        print(
            "  Note: no active prop models found. "
            "Run 'python -m src.cli train --sport nba --kind props' first.\n"
            "  Showing DK lines only.\n"
        )

    # Group by game
    by_game: dict[str, list[dict]] = defaultdict(list)
    for p in props:
        game_key = f"{p['away_team']} @ {p['home_team']}" if p["home_team"] else "Unknown game"
        by_game[game_key].append(p)

    now = datetime.now(UTC)
    print(f"\n{'─'*100}")
    print(f"  {sport.upper()} PLAYER PROPS — all lines  ({now.strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'─'*100}")

    with sync_session_factory() as session:
        for game_key in sorted(by_game):
            print(f"\n  {game_key}")
            game_props = sorted(by_game[game_key], key=lambda x: (x["stat"], x["player_name"]))

            for prop in game_props:
                player_name = prop["player_name"]
                stat = prop["stat"]
                line = prop["line"]

                model = models.get(stat)
                model_median: float | None = None
                model_over: float | None = None
                edge: float | None = None

                if model is not None:
                    fn = feat_names.get(stat, [])
                    player = _lookup_player(session, sport, player_name)
                    if player is not None:
                        team = _get_player_team(session, player.id, sport)
                        if team is not None:
                            opponent = _get_opponent(session, team.id, sport, hours)
                            if opponent is not None:
                                feats = _build_player_features_safe(
                                    session, player, team, opponent, stat, sport
                                )
                                if feats and fn:
                                    model_median = _get_median(model, feats, fn)
                                    model_over = _implied_over_prob(model, feats, fn, line)

                dk_over_implied = None
                if prop["over_odds"] is not None:
                    dk_over_implied = _american_to_implied(prop["over_odds"])

                if model_over is not None and dk_over_implied is not None:
                    edge = model_over - dk_over_implied

                _print_prop_row(
                    player_name,
                    stat,
                    line,
                    prop["over_odds"],
                    prop["under_odds"],
                    model_median,
                    model_over,
                    dk_over_implied,
                    edge,
                )

    total = len(props)
    print(f"\n{'─'*100}")
    print(f"  {total} prop line(s) shown\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--stat", default=None)
    parser.add_argument("--hours", type=int, default=36)
    args = parser.parse_args()
    main(args.sport, args.stat, args.hours)
