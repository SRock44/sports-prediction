"""Gather historical NBA training data from stats.nba.com.

Runs three phases in sequence:
  1. Teams + rosters     — static data, fast
  2. Season schedules    — one request per season type per season
  3. Box scores          — two requests per completed game (resumable)

Typical runtime for 5 seasons:
  Phase 1: ~2 min
  Phase 2: ~5 min
  Phase 3: ~4-6 hours  (1 req/sec enforced by nba_api; ~1,300 games/season)

The script is fully resumable. Re-running it skips games that already have
box scores in the database. Kill it at any time and restart safely.

Usage (inside Docker):
  docker compose exec api python scripts/gather_nba_training_data.py
  docker compose exec api python scripts/gather_nba_training_data.py --seasons 3
  docker compose exec api python scripts/gather_nba_training_data.py --skip-box-scores
  docker compose exec api python scripts/gather_nba_training_data.py --season-start 2020

Usage (local, with DB in Docker):
  DATABASE_URL_SYNC=postgresql://... python scripts/gather_nba_training_data.py
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure project root is on PYTHONPATH when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import get_logger
from src.core.time import nba_season_for_date
from src.db.models import Game, PlayerGameStats, Sport
from src.db.session import get_sync_session
from src.ingest.nba.games import ingest_season_schedule, ingest_box_scores
from src.ingest.nba.players import sync_teams, sync_players

log = get_logger(__name__)


# ── Progress helpers ───────────────────────────────────────────────────────────

class Progress:
    """Simple inline progress printer — no external deps."""

    def __init__(self, total: int, label: str, unit: str = "games") -> None:
        self.total = total
        self.label = label
        self.unit = unit
        self.done = 0
        self.errors = 0
        self.skipped = 0
        self._start = time.monotonic()

    def tick(self, *, error: bool = False, skipped: bool = False) -> None:
        self.done += 1
        if error:
            self.errors += 1
        if skipped:
            self.skipped += 1
        self._print()

    def _print(self) -> None:
        elapsed = time.monotonic() - self._start
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.done) / rate if rate > 0 else 0
        pct = self.done / self.total * 100 if self.total else 0

        bar_len = 30
        filled = int(bar_len * self.done / self.total) if self.total else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        eta = _fmt_seconds(remaining)
        print(
            f"\r  [{bar}] {pct:5.1f}%  {self.done}/{self.total} {self.unit}"
            f"  err={self.errors}  skip={self.skipped}  ETA {eta}",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        elapsed = time.monotonic() - self._start
        print(
            f"\r  {self.label} done — {self.done}/{self.total} {self.unit}"
            f"  errors={self.errors}  skipped={self.skipped}"
            f"  ({_fmt_seconds(elapsed)})" + " " * 20
        )


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


# ── Phase implementations ──────────────────────────────────────────────────────

def phase_teams_and_rosters(season_years: list[int]) -> None:
    _header("Phase 1 — Sync teams and rosters")

    with get_sync_session() as session:
        print("  Syncing teams... ", end="", flush=True)
        r = sync_teams(session)
        session.commit()
        print(f"done  +{r.rows_inserted} new  updated={r.rows_updated}")

    for season_year in season_years:
        with get_sync_session() as session:
            print(f"  Rosters for {season_year}-{str(season_year+1)[-2:]}... ", end="", flush=True)
            try:
                r = sync_players(session, season_year)
                session.commit()
                print(f"done  +{r.rows_inserted} new  updated={r.rows_updated}")
            except Exception as exc:
                print(f"WARN: {exc}")
            time.sleep(1.2)  # extra courtesy pause between roster calls


def phase_schedules(season_years: list[int]) -> dict[int, int]:
    """Ingest schedules. Returns {season_year: games_ingested}."""
    _header("Phase 2 — Ingest season schedules")
    counts: dict[int, int] = {}

    for season_year in season_years:
        label = f"{season_year}-{str(season_year+1)[-2:]}"
        print(f"  Season {label}... ", end="", flush=True)
        try:
            with get_sync_session() as session:
                r = ingest_season_schedule(session, season_year)
                session.commit()
            counts[season_year] = r.rows_inserted + r.rows_updated
            print(
                f"done  +{r.rows_inserted} new  updated={r.rows_updated}"
                + (f"  ERRORS={len(r.errors)}" if r.errors else "")
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            counts[season_year] = 0
            log.exception("schedule.ingest_failed", season=label, error=str(exc))

    return counts


def phase_box_scores(season_years: list[int]) -> None:
    """Fetch box scores for every completed game that doesn't already have them.

    Box scores are the heavy lift: traditional + advanced = 2 API calls per game
    at 1 req/sec each. The query to find pending games runs each batch so restarts
    pick up exactly where we left off.
    """
    _header("Phase 3 — Ingest box scores (resumable)")

    # Find all completed games across all target seasons that have no box score yet.
    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            print("  No NBA sport record found. Run phase 1 first.")
            return

        pending_ids: list[str] = (
            session.query(Game.external_id)
            .filter(
                Game.sport_id == sport.id,
                Game.status == "final",
                Game.season.in_(season_years),
                ~Game.id.in_(
                    session.query(PlayerGameStats.game_id).distinct()
                ),
            )
            .order_by(Game.scheduled_utc)
            .all()
        )
        pending_ids = [row[0] for row in pending_ids]

    total_completed = _count_completed(season_years)
    print(f"  Completed games in target seasons: {total_completed}")
    print(f"  Games still needing box scores:    {len(pending_ids)}")

    if not pending_ids:
        print("  Nothing to fetch — all box scores already present.")
        return

    # Estimate time
    est_seconds = len(pending_ids) * 2 * 1.2  # 2 calls × ~1.2s each
    print(f"  Estimated time: {_fmt_seconds(est_seconds)}")
    print()

    prog = Progress(len(pending_ids), "Box scores")

    for game_ext_id in pending_ids:
        try:
            with get_sync_session() as session:
                r = ingest_box_scores(session, game_ext_id)
                session.commit()

            if r.errors:
                prog.tick(error=True)
                log.warning("box_score.error", game_id=game_ext_id, errors=r.errors)
            else:
                prog.tick()

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Progress saved — restart to continue.")
            prog.finish()
            sys.exit(0)
        except Exception as exc:
            prog.tick(error=True)
            log.exception("box_score.unexpected_error", game_id=game_ext_id, error=str(exc))

    prog.finish()


def _count_completed(season_years: list[int]) -> int:
    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            return 0
        return (
            session.query(Game)
            .filter(
                Game.sport_id == sport.id,
                Game.status == "final",
                Game.season.in_(season_years),
            )
            .count()
        )


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(season_years: list[int]) -> None:
    _header("Summary")
    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            print("  No NBA data found.")
            return

        for season_year in season_years:
            label = f"{season_year}-{str(season_year+1)[-2:]}"
            total = session.query(Game).filter(
                Game.sport_id == sport.id, Game.season == season_year
            ).count()
            with_scores = (
                session.query(Game)
                .filter(
                    Game.sport_id == sport.id,
                    Game.season == season_year,
                    Game.status == "final",
                    Game.id.in_(
                        session.query(PlayerGameStats.game_id).distinct()
                    ),
                )
                .count()
            )
            print(f"  {label}  — {total:4d} games  {with_scores:4d} with box scores")

    print()
    print("  Next steps:")
    print("    Build features:   docker compose exec api python scripts/build_nba_features.py")
    print("    Train model:      docker compose exec api python -m src.cli train --sport nba --kind winner --promote")
    print("    Score upcoming:   docker compose exec api python -m src.cli score --sport nba")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather historical NBA training data from stats.nba.com"
    )
    parser.add_argument(
        "--seasons", type=int, default=5,
        help="Number of past seasons to gather (default: 5)"
    )
    parser.add_argument(
        "--season-start", type=int, default=None,
        help="Oldest season year to include (e.g. 2020). Overrides --seasons."
    )
    parser.add_argument(
        "--skip-rosters", action="store_true",
        help="Skip roster sync (faster if you only need game data)"
    )
    parser.add_argument(
        "--skip-box-scores", action="store_true",
        help="Only ingest schedules, skip box score fetching"
    )
    parser.add_argument(
        "--box-scores-only", action="store_true",
        help="Skip phases 1+2, only fetch missing box scores"
    )
    args = parser.parse_args()

    today = date.today()
    current_season = nba_season_for_date(today)

    if args.season_start is not None:
        season_start = args.season_start
    else:
        season_start = current_season - args.seasons + 1

    season_years = list(range(season_start, current_season + 1))

    print(f"\nNBA Training Data Gatherer")
    print(f"Target seasons: {[f'{y}-{str(y+1)[-2:]}' for y in season_years]}")
    print(f"Source:         stats.nba.com (free, unofficial endpoint)")
    print(f"Rate limit:     1 req/sec (enforced)")
    print()

    start = time.monotonic()

    if args.box_scores_only:
        phase_box_scores(season_years)
    else:
        if not args.skip_rosters:
            phase_teams_and_rosters(season_years)
        else:
            _header("Phase 1 — Skipped (--skip-rosters)")

        phase_schedules(season_years)

        if not args.skip_box_scores:
            phase_box_scores(season_years)
        else:
            _header("Phase 3 — Skipped (--skip-box-scores)")

    print_summary(season_years)
    print(f"\nTotal elapsed: {_fmt_seconds(time.monotonic() - start)}")


if __name__ == "__main__":
    main()
