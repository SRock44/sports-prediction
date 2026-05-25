"""Fetch and display DraftKings player prop odds for today's games.

Hits DK's Nash API directly — no key required.
Useful for verifying what lines DK is offering before running props_all / props_winners.

Usage:
  python -m scripts.dk_props              # NBA
  python -m scripts.dk_props --sport mlb
  python -m scripts.dk_props --stat PTS   # filter to one stat
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from src.ingest.odds.draftkings import get_player_props


def _fmt_odds(v: float | None) -> str:
    if v is None:
        return "  n/a"
    return f"{int(v):+5d}"


def main(sport: str = "nba", stat_filter: str | None = None) -> None:
    print(f"\nFetching DraftKings {sport.upper()} player props…")
    props = get_player_props(sport)

    if not props:
        print("  No player prop lines found. The category may not be live yet.")
        return

    if stat_filter:
        props = [p for p in props if p["stat"].upper() == stat_filter.upper()]
        if not props:
            print(f"  No {stat_filter.upper()} props found.")
            return

    # Group by stat → then by game
    by_stat: dict[str, list[dict]] = defaultdict(list)
    for p in props:
        by_stat[p["stat"]].append(p)

    total = 0
    for stat in sorted(by_stat):
        stat_props = by_stat[stat]
        # Group by game (away @ home)
        by_game: dict[str, list[dict]] = defaultdict(list)
        for p in stat_props:
            game_key = f"{p['away_team']} @ {p['home_team']}" if p["home_team"] else "Unknown game"
            by_game[game_key].append(p)

        print(f"\n  ── {stat} ──")
        for game_key in sorted(by_game):
            print(f"    {game_key}")
            for p in sorted(by_game[game_key], key=lambda x: x["player_name"]):
                over_str = _fmt_odds(p["over_odds"])
                under_str = _fmt_odds(p["under_odds"])
                print(
                    f"      {p['player_name']:<25}  O/U {p['line']:5.1f}"
                    f"  over {over_str}  under {under_str}"
                )
                total += 1

    print(f"\n  Total: {total} prop line(s) across {len(by_stat)} stat(s)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--stat", default=None, help="Filter to one stat (PTS, REB, AST, etc.)")
    args = parser.parse_args()
    main(args.sport, args.stat)
