"""Pull 5 seasons of MLB historical training data from statsapi.mlb.com.

Phases:
  1. Teams + rosters
  2. Season schedules
  3. Box scores for all completed games (resumable)

Usage:
  docker compose run --rm api python scripts/gather_mlb_training_data.py --seasons 5
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.models import Game, PlayerGameStats, Sport, Team
from src.db.session import get_sync_session
from src.ingest.mlb.games import ingest_box_score, ingest_season_schedule, sync_teams
from src.ingest.mlb.players import sync_roster


def _current_mlb_season() -> int:
    today = date.today()
    # MLB season: April-October; treat Jan-Mar as still "last season" for default
    return today.year if today.month >= 4 else today.year - 1


def _progress(done: int, total: int, errors: int, skipped: int, eta_secs: float | None) -> str:
    pct = done / total if total else 0
    filled = int(pct * 30)
    bar = "█" * filled + "░" * (30 - filled)
    eta = f"ETA {eta_secs / 3600:.1f}h" if eta_secs else "ETA --"
    return f"  [{bar}] {pct*100:5.1f}%  {done}/{total} games  err={errors}  skip={skipped}  {eta}"


def phase1_teams(session, seasons: list[int]) -> None:
    print("\nPhase 1 — Sync teams and rosters")
    result = sync_teams(session, seasons[-1])
    session.commit()
    print(f"  Teams synced  +{result.rows_inserted} new  updated={result.rows_updated}")

    sport = session.query(Sport).filter_by(code="mlb").first()
    teams = session.query(Team).filter_by(sport_id=sport.id).all()

    for season in seasons:
        label = str(season)
        print(f"  Rosters for {label}... ", end="", flush=True)
        ok = err = 0
        for team in teams:
            try:
                sync_roster(session, team.external_id, season)
                ok += 1
            except Exception:
                err += 1
        session.commit()
        print(f"done  ok={ok}  err={err}")


def phase2_schedules(session, seasons: list[int]) -> None:
    print("\nPhase 2 — Ingest season schedules")
    for season in seasons:
        print(f"  Schedule {season}... ", end="", flush=True)
        try:
            result = ingest_season_schedule(session, season)
            session.commit()
            print(f"done  +{result.rows_inserted} games  updated={result.rows_updated}  err={len(result.errors)}")
        except Exception as e:
            print(f"FAILED: {e}")


def phase3_box_scores(session, seasons: list[int]) -> None:
    print("\nPhase 3 — Fetch box scores")
    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        print("  No MLB sport found — run Phase 1 first.")
        return

    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.status == "final",
            Game.season.in_(seasons),
        )
        .order_by(Game.scheduled_utc)
        .all()
    )

    # Find which games already have box scores
    games_with_bs = set(
        row[0]
        for row in session.query(PlayerGameStats.game_id)
        .filter(PlayerGameStats.game_id.in_([g.id for g in games]))
        .distinct()
        .all()
    )

    pending = [g for g in games if g.id not in games_with_bs]
    total = len(games)
    done = total - len(pending)
    errors = 0
    skipped = 0

    print(f"  {total} completed games total  {done} already have box scores  {len(pending)} to fetch")
    if not pending:
        print("  All box scores already present.")
        return

    start_time = time.monotonic()

    for i, game in enumerate(pending):

        try:
            result = ingest_box_score(session, game.external_id)
            session.commit()
            if result.errors:
                errors += 1
        except Exception:
            errors += 1
            session.rollback()

        done += 1
        elapsed = time.monotonic() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(pending) - (i + 1)
        eta = remaining / rate if rate > 0 else None

        print(
            "\r" + _progress(done, total, errors, skipped, eta),
            end="",
            flush=True,
        )

    print()  # newline after progress bar


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather MLB training data")
    parser.add_argument("--seasons", type=int, default=5, help="Number of past seasons to backfill")
    parser.add_argument("--season-start", type=int, default=None, help="Override starting season year")
    parser.add_argument("--skip-rosters", action="store_true")
    parser.add_argument("--skip-box-scores", action="store_true")
    parser.add_argument("--box-scores-only", action="store_true")
    args = parser.parse_args()

    current = _current_mlb_season()
    start_year = args.season_start or (current - args.seasons + 1)
    seasons = list(range(start_year, current + 1))

    print("MLB Training Data Gatherer")
    print(f"Target seasons: {seasons}")
    print("Source:         statsapi.mlb.com (free, official endpoint)")
    print("Rate limit:     ~0.3s between requests\n")

    with get_sync_session() as session:
        if not args.box_scores_only:
            if not args.skip_rosters:
                phase1_teams(session, seasons)
            phase2_schedules(session, seasons)

        if not args.skip_box_scores:
            phase3_box_scores(session, seasons)

    print("\nDone.")


if __name__ == "__main__":
    main()
