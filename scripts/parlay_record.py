"""Show Discord parlay record — wins, losses, and pending parlays.

Reads from the discord_parlays table which tracks every parlay a user locks
in via the /predict Discord bot command.

Usage:
  python -m scripts.parlay_record              # all sports
  python -m scripts.parlay_record --sport nba
  python -m scripts.parlay_record --sport mlb
  python -m scripts.parlay_record --legs 3     # filter to 3-leg parlays
  python -m scripts.parlay_record --recent 30  # last N days (default: all time)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from src.db.session import sync_session_factory


def _pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d > 0 else "  n/a"


def main(
    sport: str | None = None,
    n_legs: int | None = None,
    recent_days: int | None = None,
) -> None:
    filters = ["1=1"]
    params: dict = {}

    if sport:
        filters.append("sport_code = :sport")
        params["sport"] = sport.lower()
    if n_legs:
        filters.append("n_legs = :n_legs")
        params["n_legs"] = n_legs
    if recent_days:
        cutoff = datetime.now(UTC) - timedelta(days=recent_days)
        filters.append("created_at >= :cutoff")
        params["cutoff"] = cutoff

    where = " AND ".join(filters)

    with sync_session_factory() as s:
        rows = s.execute(
            text(f"""
            SELECT
                id,
                discord_username,
                sport_code,
                n_legs,
                parlay_odds_american,
                parlay_ev,
                status,
                n_correct,
                legs,
                created_at,
                resolved_at
            FROM discord_parlays
            WHERE {where}
            ORDER BY created_at DESC
            """),
            params,
        ).fetchall()

    if not rows:
        print("\n  No parlay records found.\n")
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    total = len(rows)
    won = sum(1 for r in rows if r.status == "won")
    lost = sum(1 for r in rows if r.status == "lost")
    pending = sum(1 for r in rows if r.status == "pending")
    partial = sum(1 for r in rows if r.status == "partial")
    settled = won + lost + partial

    # Per-leg breakdown
    by_legs: dict[int, dict[str, int]] = defaultdict(lambda: {"won": 0, "lost": 0, "pending": 0, "partial": 0})
    for r in rows:
        by_legs[r.n_legs][r.status] = by_legs[r.n_legs].get(r.status, 0) + 1

    # Total legs correct out of settled legs
    total_legs_correct = sum(r.n_correct for r in rows if r.status != "pending")
    total_legs_settled = sum(
        r.n_legs for r in rows if r.status != "pending"
    )

    now = datetime.now(UTC)
    scope = sport.upper() if sport else "ALL SPORTS"
    period = f"last {recent_days}d" if recent_days else "all time"

    print(f"\n{'═'*60}")
    print(f"  PARLAY RECORD  —  {scope}  ({period})")
    print(f"{'═'*60}")
    print(f"  Total parlays : {total}")
    print(f"  Won           : {won}  ({_pct(won, settled)} of settled)")
    print(f"  Lost          : {lost}")
    print(f"  Partial       : {partial}")
    print(f"  Pending       : {pending}")
    if total_legs_settled > 0:
        print(f"  Leg accuracy  : {total_legs_correct}/{total_legs_settled}  ({_pct(total_legs_correct, total_legs_settled)})")

    if by_legs:
        print(f"\n  By parlay size:")
        for nl in sorted(by_legs):
            bl = by_legs[nl]
            w = bl.get("won", 0)
            l = bl.get("lost", 0)
            p = bl.get("pending", 0)
            s = w + l + bl.get("partial", 0)
            print(f"    {nl}-leg: {w}W {l}L {p}P  ({_pct(w, s)} win rate)")

    # ── Recent parlays ────────────────────────────────────────────────────────
    display_n = min(20, len(rows))
    print(f"\n  Recent parlays (showing {display_n} of {total}):")
    print(f"  {'─'*56}")
    print(f"  {'DATE':<12} {'USER':<16} {'SP':>3} {'LEGS':>4} {'STATUS':>8} {'CORRECT':>8} {'ODDS':>7}")
    print(f"  {'─'*56}")

    status_icon = {"won": "✓ WON", "lost": "✗ LOST", "pending": "… PEND", "partial": "~ PART"}

    for r in rows[:display_n]:
        created = r.created_at
        if hasattr(created, "tzinfo") and created.tzinfo:
            date_str = created.astimezone(UTC).strftime("%m/%d %H:%M")
        else:
            date_str = created.strftime("%m/%d %H:%M") if created else "?"

        odds_str = f"{int(r.parlay_odds_american):+d}" if r.parlay_odds_american else "  n/a"
        correct_str = f"{r.n_correct}/{r.n_legs}"
        icon = status_icon.get(r.status, r.status)
        user = (r.discord_username or "unknown")[:15]

        print(
            f"  {date_str:<12} {user:<16} {r.sport_code.upper():>3}"
            f" {r.n_legs:>4}  {icon:>8}  {correct_str:>7}  {odds_str:>7}"
        )

    print(f"  {'─'*56}")
    print(f"  As of {now.strftime('%Y-%m-%d %H:%M UTC')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default=None, help="nba or mlb (default: all)")
    parser.add_argument("--legs", type=int, default=None, help="Filter to N-leg parlays")
    parser.add_argument("--recent", type=int, default=None, help="Last N days (default: all time)")
    args = parser.parse_args()
    main(args.sport, args.legs, args.recent)
