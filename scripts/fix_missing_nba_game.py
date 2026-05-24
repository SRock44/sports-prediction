"""Fetch CLE vs NYK result from ESPN API and upsert game 21786.

ESPN's public scoreboard endpoint is not blocked by nba.com CDN.
Run: docker compose exec worker python scripts/fix_missing_nba_game.py
"""

from __future__ import annotations

import json
import urllib.request

from sqlalchemy import text

from src.db.session import sync_session_factory
from src.tasks.outcome_tasks import check_outcomes

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20260523"


def fetch_espn_games() -> list[dict]:
    req = urllib.request.Request(ESPN_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["events"]


def main() -> None:
    print("Fetching ESPN NBA scoreboard for 2026-05-23...")
    events = fetch_espn_games()
    print(f"Found {len(events)} events")

    target = None
    for ev in events:
        comps = ev.get("competitions", [{}])[0]
        teams = {c["homeAway"]: c["team"]["displayName"] for c in comps.get("competitors", [])}
        home = teams.get("home", "")
        away = teams.get("away", "")
        print(f"  {away} @ {home}")
        if "Cavaliers" in home and "Knicks" in away:
            target = (ev, comps)
            break

    if target is None:
        print("ERROR: CLE vs NYK not found in ESPN data for 2026-05-23")
        return

    ev, comps = target
    competitors = comps.get("competitors", [])
    scores: dict[str, int] = {}
    for c in competitors:
        scores[c["homeAway"]] = int(c.get("score", 0))

    home_score = scores.get("home", 0)
    away_score = scores.get("away", 0)
    status_text = comps.get("status", {}).get("type", {}).get("description", "").lower()
    is_final = "final" in status_text or comps.get("status", {}).get("type", {}).get(
        "completed", False
    )

    print(
        f"\nResult: CLE {home_score} - NYK {away_score}  |  status={status_text}  final={is_final}"
    )

    if not is_final:
        print("WARNING: game not marked final yet — skipping DB update")
        return

    with sync_session_factory() as session:
        # Get sport/team IDs
        sport_id = session.execute(text("SELECT id FROM sports WHERE code='nba'")).scalar()
        cle_id = session.execute(
            text("SELECT id FROM teams WHERE name ILIKE '%cavalier%' AND sport_id=:s"),
            {"s": sport_id},
        ).scalar()
        nyk_id = session.execute(
            text("SELECT id FROM teams WHERE name ILIKE '%knick%' AND sport_id=:s"), {"s": sport_id}
        ).scalar()

        print(f"sport_id={sport_id}  CLE team_id={cle_id}  NYK team_id={nyk_id}")

        # Check if game 21786 exists
        existing = session.execute(text("SELECT id, status FROM games WHERE id=21786")).first()

        if existing:
            print(f"Game 21786 exists with status={existing.status} — updating to final")
            session.execute(
                text("""
                UPDATE games
                SET status='final', home_score=:hs, away_score=:as
                WHERE id=21786
            """),
                {"hs": home_score, "as": away_score},
            )
        else:
            print("Game 21786 missing — inserting")
            session.execute(
                text("""
                INSERT INTO games (id, sport_id, home_team_id, away_team_id,
                                   scheduled_utc, status, home_score, away_score,
                                   external_id)
                VALUES (21786, :sport_id, :home_id, :away_id,
                        '2026-05-24 00:10:00+00', 'final', :hs, :as,
                        'espn_cle_nyk_20260523')
            """),
                {
                    "sport_id": sport_id,
                    "home_id": cle_id,
                    "away_id": nyk_id,
                    "hs": home_score,
                    "as": away_score,
                },
            )

        # Advance the sequence past 21786 if needed
        session.execute(
            text("""
            SELECT setval(pg_get_serial_sequence('games','id'),
                          GREATEST(nextval(pg_get_serial_sequence('games','id')), 21787))
        """)
        )

        session.commit()
        print("Game upserted. Running check_outcomes_nba...")

    result = check_outcomes("nba")
    print(f"check_outcomes result: {result}")


if __name__ == "__main__":
    main()
