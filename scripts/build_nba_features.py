"""Build matchup features for all completed NBA games that lack them.

Run this after gather_nba_training_data.py finishes. Features are stored in
`matchup_features` (one row per game) and are what the model actually trains on.

Resumable: games that already have a matchup_features row are skipped unless
--force is passed.

Usage:
  docker compose exec api python scripts/build_nba_features.py
  docker compose exec api python scripts/build_nba_features.py --seasons 3
  docker compose exec api python scripts/build_nba_features.py --force   # rebuild all
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import or_

from src.core.logging import get_logger
from src.core.time import as_of_for_game, nba_season_for_date
from src.db.models import Game, MatchupFeature, PlayerGameStats, Sport
from src.db.session import get_sync_session
from src.features.nba.matchup import build_matchup_features

log = get_logger(__name__)


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def _header(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def build_features_for_seasons(season_years: list[int], force: bool = False) -> None:
    _header("Building NBA matchup features")

    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            print("  ERROR: No NBA sport record. Run gather_nba_training_data.py first.")
            sys.exit(1)

        # Regular season games only (exclude preseason 'PR'); must have box scores
        base_q = (
            session.query(Game.id, Game.external_id, Game.scheduled_utc, Game.season)
            .filter(
                Game.sport_id == sport.id,
                Game.status == "final",
                Game.season.in_(season_years),
                or_(Game.meta["game_type"] is None, Game.meta["game_type"].astext != "PR"),
                Game.id.in_(session.query(PlayerGameStats.game_id).distinct()),
            )
            .order_by(Game.scheduled_utc)
        )

        if not force:
            already_built = session.query(MatchupFeature.game_id).subquery()
            base_q = base_q.filter(~Game.id.in_(already_built))

        pending = base_q.all()

    total = len(pending)
    if total == 0:
        print("  All games already have features. Use --force to rebuild.")
        return

    print(f"  Games to process: {total}")
    print(f"  Estimated time:   {_fmt_seconds(total * 0.05)} (fast — pure DB reads)")
    print()

    done = errors = skipped = 0
    start = time.monotonic()

    for game_id, game_ext_id, scheduled_utc, season in pending:
        try:
            as_of = as_of_for_game(scheduled_utc)

            with get_sync_session() as session:
                # Re-fetch the full Game ORM object for the feature builder
                game = session.get(Game, game_id)
                if game is None:
                    skipped += 1
                    continue

                features = build_matchup_features(session, game, as_of)
                if features is None or not features:
                    skipped += 1
                    continue

                existing = session.query(MatchupFeature).filter_by(game_id=game_id).first()
                if existing is None:
                    session.add(
                        MatchupFeature(
                            game_id=game_id,
                            features=features,
                            computed_at=scheduled_utc,
                        )
                    )
                else:
                    existing.features = features
                    existing.computed_at = scheduled_utc

                session.commit()
                done += 1

        except KeyboardInterrupt:
            print(f"\n\n  Interrupted at game {game_ext_id}. Restart to continue from here.")
            _print_progress(done, errors, skipped, total, start)
            sys.exit(0)
        except Exception as exc:
            errors += 1
            log.warning("features.build_failed", game=game_ext_id, season=season, error=str(exc))

        # Print progress every 50 games
        if (done + errors + skipped) % 50 == 0:
            _print_progress(done, errors, skipped, total, start, inline=True)

    print()  # newline after inline progress
    _print_progress(done, errors, skipped, total, start)
    print()

    # Final stats
    _header("Feature build complete")
    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        for season_year in season_years:
            total_games = (
                session.query(Game)
                .filter(
                    Game.sport_id == sport.id, Game.season == season_year, Game.status == "final"
                )
                .count()
            )
            with_features = (
                session.query(MatchupFeature)
                .join(Game, MatchupFeature.game_id == Game.id)
                .filter(Game.sport_id == sport.id, Game.season == season_year)
                .count()
            )
            label = f"{season_year}-{str(season_year + 1)[-2:]}"
            print(f"  {label}  — {total_games:4d} games  {with_features:4d} with features")

    print()
    print("  Ready to train:")
    print("    python -m src.cli train --sport nba --kind winner --trials 50 --promote")


def _print_progress(
    done: int, errors: int, skipped: int, total: int, start: float, inline: bool = False
) -> None:
    elapsed = time.monotonic() - start
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done - errors - skipped) / rate if rate > 0 else 0
    pct = (done + errors + skipped) / total * 100 if total else 0

    bar_len = 30
    filled = int(bar_len * (done + errors + skipped) / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    msg = (
        f"\r  [{bar}] {pct:5.1f}%  built={done}  err={errors}  skip={skipped}"
        f"  ETA {_fmt_seconds(remaining)}"
    )
    if inline:
        print(msg, end="", flush=True)
    else:
        print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NBA matchup features for completed historical games"
    )
    parser.add_argument(
        "--seasons", type=int, default=5, help="Number of past seasons (default: 5)"
    )
    parser.add_argument(
        "--season-start", type=int, default=None, help="Oldest season year (e.g. 2020)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild features even if they already exist"
    )
    args = parser.parse_args()

    today = date.today()
    current_season = nba_season_for_date(today)
    season_start = args.season_start if args.season_start else current_season - args.seasons + 1
    season_years = list(range(season_start, current_season + 1))

    print("\nNBA Feature Builder")
    print(f"Target seasons: {[f'{y}-{str(y + 1)[-2:]}' for y in season_years]}")
    print(f"Force rebuild:  {args.force}")

    build_features_for_seasons(season_years, force=args.force)


if __name__ == "__main__":
    main()
