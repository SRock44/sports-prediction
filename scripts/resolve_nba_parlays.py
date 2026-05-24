"""Resolve pending discord parlays for game 21786 (CLE vs NYK, NYK won 121-108)."""

from __future__ import annotations

import json

from sqlalchemy import text

from src.db.session import sync_session_factory

with sync_session_factory() as session:
    parlays = session.execute(
        text(
            "SELECT id, legs FROM discord_parlays WHERE status='pending' AND legs::text LIKE '%21786%'"
        )
    ).fetchall()
    print(f"Found {len(parlays)} pending parlays referencing game 21786")

    for row in parlays:
        legs = row.legs
        for leg in legs:
            if leg.get("game_id") == 21786:
                leg["result"] = "lost"  # pick was CLE home, NYK away won
                leg["actual_winner"] = "away"
        n_correct = sum(1 for leg in legs if leg.get("result") == "won")
        status = "won" if n_correct == len(legs) else "lost"
        legs_json = json.dumps(legs)
        session.execute(
            text("""
                UPDATE discord_parlays
                SET legs = cast(:legs AS jsonb),
                    status = :status,
                    n_correct = :nc
                WHERE id = :id
            """),
            {"legs": legs_json, "status": status, "nc": n_correct, "id": row.id},
        )
        print(f"  Parlay {row.id} -> {status}  ({n_correct}/{len(legs)} legs correct)")

    session.commit()
    print("Done.")
