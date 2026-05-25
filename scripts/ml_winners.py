"""Show model moneyline picks that clear the edge + confidence threshold (like /kirkova).

Only shows games where the model has meaningful edge over the DK implied probability.
Sorted best edge first.

Usage:
  python -m scripts.ml_winners              # NBA
  python -m scripts.ml_winners --sport mlb
  python -m scripts.ml_winners --sport nba --n 10
"""

from __future__ import annotations

import argparse

from src.db.session import sync_session_factory
from src.models.parlay import select_top_picks


def _fmt_odds(v: int) -> str:
    return f"{v:+d}"


def main(sport: str = "nba", n: int = 10, bookmaker: str = "draftkings") -> None:
    with sync_session_factory() as s:
        picks = select_top_picks(s, sport, bookmaker=bookmaker, n=n)

    if not picks:
        print(
            f"\nNo qualifying {sport.upper()} picks right now (edge/confidence thresholds not met).\n"
        )
        return

    from datetime import UTC, datetime

    now = datetime.now(UTC)

    print(f"\n{'─' * 72}")
    print(
        f"  {sport.upper()} TOP PICKS — edge filter applied  ({now.strftime('%Y-%m-%d %H:%M UTC')})"
    )
    print(f"{'─' * 72}")
    header = f"  {'PICK':<28} {'PROB':>6}  {'IMPLIED':>7}  {'EDGE':>6}  {'ODDS':>6}"
    print(header)
    print(f"{'─' * 72}")

    for p in picks:
        pick_team = p.home_team if p.pick == "home" else p.away_team
        pick_short = pick_team.split()[-1]
        matchup = f"{p.away_team.split()[-1]} @ {p.home_team.split()[-1]}"
        label = f"{pick_short} ({matchup})"
        print(
            f"  {label:<28} {p.model_prob:>6.1%}  {p.implied_prob:>7.1%}"
            f"  {p.edge:>+6.1%}  {_fmt_odds(p.odds_american):>6}"
        )

    print(f"{'─' * 72}")
    print(f"  {len(picks)} qualifying pick(s)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--bookmaker", default="draftkings")
    args = parser.parse_args()
    main(args.sport, args.n, args.bookmaker)
