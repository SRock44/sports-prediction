"""Show all moneyline model predictions for upcoming games (no edge filter).

Usage:
  python -m scripts.ml_all              # NBA (default)
  python -m scripts.ml_all --sport mlb
  python -m scripts.ml_all --sport nba --hours 48
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from src.db.session import sync_session_factory


def _fmt_odds(v: float | None) -> str:
    if v is None:
        return "  n/a "
    return f"{int(v):+5d}" if v != 0 else " even"


def main(sport: str = "nba", hours: int = 36) -> None:
    with sync_session_factory() as s:
        rows = s.execute(
            text("""
            SELECT
                g.id           AS game_id,
                ht.name        AS home_team,
                at.name        AS away_team,
                g.scheduled_utc,
                g.status,
                p.probability  AS home_prob,
                go.home_price  AS dk_home_ml,
                go.away_price  AS dk_away_ml,
                go.fetched_at  AS odds_snap
            FROM predictions p
            JOIN games       g  ON g.id = p.game_id
            JOIN sports      sp ON sp.id = g.sport_id
            JOIN teams       ht ON ht.id = g.home_team_id
            JOIN teams       at ON at.id = g.away_team_id
            LEFT JOIN game_odds go
                   ON go.game_id = g.id
                  AND go.bookmaker = 'draftkings'
                  AND go.market    = 'h2h'
            WHERE sp.code     = :sport
              AND p.target     = 'home_won'
              AND p.player_id IS NULL
              AND g.scheduled_utc BETWEEN NOW() - INTERVAL '4 hours'
                                      AND NOW() + :hrs * INTERVAL '1 hour'
            ORDER BY g.scheduled_utc, go.fetched_at DESC NULLS LAST
            """),
            {"sport": sport, "hrs": hours},
        ).fetchall()

    # Deduplicate: keep one row per game (latest odds snapshot)
    seen: set[int] = set()
    games: list = []
    for r in rows:
        if r.game_id not in seen:
            seen.add(r.game_id)
            games.append(r)

    if not games:
        print(f"No {sport.upper()} predictions found for the next {hours}h.")
        return

    now = datetime.now(UTC)
    print(f"\n{'─'*78}")
    print(f"  {sport.upper()} MONEYLINE PREDICTIONS — all games  "
          f"({now.strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'─'*78}")
    header = f"  {'MATCHUP':<35} {'TIME':>8}  {'HOME%':>6}  {'PICK':>4}  {'DK HOME':>7}  {'DK AWAY':>7}"
    print(header)
    print(f"{'─'*78}")

    for r in games:
        home_prob = float(r.home_prob)
        away_prob = 1.0 - home_prob
        pick_label = r.home_team.split()[-1] if home_prob >= 0.5 else r.away_team.split()[-1]

        # Convert scheduled_utc (may be offset-aware or naive) to local-ish display
        sched = r.scheduled_utc
        if hasattr(sched, "tzinfo") and sched.tzinfo is not None:
            sched_str = sched.astimezone(UTC).strftime("%I:%M%p")
        else:
            sched_str = sched.strftime("%I:%M%p") if sched else "  ?   "

        matchup = f"{r.away_team.split()[-1]} @ {r.home_team.split()[-1]}"
        print(
            f"  {matchup:<35} {sched_str:>8}  {home_prob:>5.1%}  {pick_label:>4}"
            f"  {_fmt_odds(r.dk_home_ml):>7}  {_fmt_odds(r.dk_away_ml):>7}"
        )

    print(f"{'─'*78}")
    print(f"  {len(games)} game(s)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--hours", type=int, default=36)
    args = parser.parse_args()
    main(args.sport, args.hours)
