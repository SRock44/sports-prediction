"""Backfill MLB starting pitcher form features in matchup_features.

Problem: matchup_features for historical games were built *before* games happened,
so _get_confirmed_starter() found no SP (box score not yet ingested). All games
have home_sp_form_known=0 and default ERA/K%/BB% values, giving the model zero
pitcher signal.

Fix: now that games are final and box scores are ingested, re-identify the SP for
each game and compute their rolling form from PRIOR starts (as_of = game start time,
so the current game's own stats are excluded — no leakage).

Only patches sp_form_* keys in the JSONB; leaves all other features unchanged.

Usage:
    python -m scripts.backfill_mlb_sp_features [--batch 500] [--dry-run]
    python -m scripts.backfill_mlb_sp_features --season 2025  # one season only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta

from sqlalchemy import text

from src.core.time import as_of_for_game
from src.db.session import sync_session_factory
from src.features.mlb.matchup import _get_confirmed_starter, _sp_rolling_form


def _patch_game(session, game_row, dry_run: bool) -> bool:
    """Identify SP from box score; compute rolling form; patch matchup_features. Returns True if patched."""
    game_id = game_row.id
    home_team_id = game_row.home_team_id
    away_team_id = game_row.away_team_id
    as_of = as_of_for_game(game_row.scheduled_utc)

    home_sp = _get_confirmed_starter(session, game_id, home_team_id, as_of)
    away_sp = _get_confirmed_starter(session, game_id, away_team_id, as_of)

    home_sp_id = (home_sp.get("playerId") or home_sp.get("player_id")) if home_sp else None
    away_sp_id = (away_sp.get("playerId") or away_sp.get("player_id")) if away_sp else None

    if home_sp_id is None and away_sp_id is None:
        return False  # no box score data at all for this game

    home_form = _sp_rolling_form(session, home_sp_id, as_of, prefix="home_sp")
    away_form = _sp_rolling_form(session, away_sp_id, as_of, prefix="away_sp")

    # Only bother if at least one SP was identified
    if not home_form.get("home_sp_form_known") and not away_form.get("away_sp_form_known"):
        return False

    sp_form_era_diff = home_form.get("home_sp_form_era", 4.50) - away_form.get("away_sp_form_era", 4.50)
    sp_form_k_pct_diff = home_form.get("home_sp_form_k_pct", 0.22) - away_form.get("away_sp_form_k_pct", 0.22)

    patch = {
        **home_form,
        **away_form,
        "sp_form_era_diff": sp_form_era_diff,
        "sp_form_k_pct_diff": sp_form_k_pct_diff,
    }

    if dry_run:
        return True

    # Merge patch into existing JSONB (|| operator in postgres)
    import json
    session.execute(
        text("""
            UPDATE matchup_features
            SET features = features || CAST(:patch AS jsonb)
            WHERE game_id = :gid
        """),
        {"gid": game_id, "patch": json.dumps(patch)},
    )
    return True


def main(batch_size: int = 500, dry_run: bool = False, season: int | None = None) -> None:
    print(f"\nMLB SP feature backfill {'(DRY RUN) ' if dry_run else ''}starting…")

    with sync_session_factory() as session:
        # Load final MLB games that have matchup_features but no SP form data
        where_season = "AND g.season = :season" if season else ""
        rows = session.execute(
            text(f"""
                SELECT g.id, g.home_team_id, g.away_team_id, g.scheduled_utc, g.season
                FROM games g
                JOIN sports sp ON sp.id = g.sport_id
                JOIN matchup_features mf ON mf.game_id = g.id
                WHERE sp.code = 'mlb'
                  AND g.status = 'final'
                  AND g.home_score IS NOT NULL
                  AND (mf.features->>'home_sp_form_known' IS NULL
                       OR (mf.features->>'home_sp_form_known')::int = 0)
                {where_season}
                ORDER BY g.scheduled_utc
            """),
            {"season": season} if season else {},
        ).fetchall()

    total = len(rows)
    if total == 0:
        print("  Nothing to backfill — all games already have SP form data.")
        return

    print(f"  Games to patch: {total}")
    patched = 0
    skipped = 0
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        with sync_session_factory() as session:
            for row in batch:
                try:
                    updated = _patch_game(session, row, dry_run)
                    if updated:
                        patched += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    print(f"  ERROR game {row.id}: {exc}")
                    skipped += 1

            if not dry_run:
                session.commit()

        done = i + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"  [{done:>5}/{total}]  patched={patched}  skipped={skipped}"
            f"  {rate:.1f} games/s  ETA {eta/60:.1f}m"
        )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s — patched {patched}, skipped {skipped} out of {total}")
    if dry_run:
        print("  DRY RUN: no changes committed.")
    else:
        print("  Backfill complete. Run 'train_challenger mlb' to retrain with pitcher features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    main(args.batch, args.dry_run, args.season)
