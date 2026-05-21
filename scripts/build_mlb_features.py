"""Build matchup features for all completed MLB games that have box scores.

Reads games from the DB, calls build_matchup_features, writes to matchup_features table.
Resumable: skips games that already have features unless --force is passed.

Usage:
  docker compose run --rm api python scripts/build_mlb_features.py
"""
from __future__ import annotations

import sys
import argparse
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.core.time import utc_now, as_of_for_game
from src.db.models import Game, MatchupFeature, PlayerGameStats, Sport
from src.db.session import get_sync_session
from src.features.mlb.matchup import build_matchup_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB matchup features")
    parser.add_argument("--force", action="store_true", help="Rebuild features even if already present")
    args = parser.parse_args()

    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="mlb").first()
        if sport is None:
            print("No MLB data found. Run gather_mlb_training_data.py first.")
            sys.exit(1)

        # Regular season games only (exclude spring training 'S', postseason 'F'/'D'/'L'/'W')
        games_with_bs = set(
            row[0]
            for row in session.query(PlayerGameStats.game_id)
            .join(Game, Game.id == PlayerGameStats.game_id)
            .filter(
                Game.sport_id == sport.id,
                Game.status == "final",
                Game.meta["game_type"].astext == "R",
            )
            .distinct()
            .all()
        )

        if not args.force:
            already_done = set(
                row[0]
                for row in session.query(MatchupFeature.game_id)
                .join(Game, Game.id == MatchupFeature.game_id)
                .filter(Game.sport_id == sport.id)
                .all()
            )
            to_process = sorted(games_with_bs - already_done)
        else:
            to_process = sorted(games_with_bs)

        total = len(to_process)
        if total == 0:
            print("All features already built. Use --force to rebuild.")
            return

        print(f"Building features for {total} MLB games...")
        errors = done = 0
        start = time.monotonic()

        for game_id in to_process:
            game = session.query(Game).get(game_id)
            if game is None:
                continue

            as_of = as_of_for_game(game.scheduled_utc)
            try:
                features = build_matchup_features(session=session, game=game, as_of=as_of)
            except Exception as e:
                errors += 1
                done += 1
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else None
                eta_str = f"ETA {eta/3600:.1f}h" if eta else "ETA --"
                print(f"\r  {done}/{total}  err={errors}  {eta_str}  [last error: {e}]", end="", flush=True)
                session.rollback()
                continue

            existing = session.query(MatchupFeature).filter_by(game_id=game_id).first()
            if existing is None:
                session.add(MatchupFeature(
                    game_id=game_id,
                    features=features,
                    computed_at=utc_now(),
                ))
            else:
                existing.features = features
                existing.computed_at = utc_now()

            session.commit()
            done += 1

            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else None
            eta_str = f"ETA {eta/3600:.1f}h" if eta else "ETA --"
            pct = done / total
            filled = int(pct * 30)
            bar = "█" * filled + "░" * (30 - filled)
            print(f"\r  [{bar}] {pct*100:5.1f}%  {done}/{total}  err={errors}  {eta_str}", end="", flush=True)

        print(f"\nDone. Built {done - errors}/{total} features  ({errors} errors)")


if __name__ == "__main__":
    main()
