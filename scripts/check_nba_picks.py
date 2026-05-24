"""Check if NBA predictions are available with odds for /kirkova and daily picks."""

from sqlalchemy import text

from src.db.session import sync_session_factory
from src.models.parlay import select_top_picks

with sync_session_factory() as s:
    # Raw check: NBA predictions joined with odds
    rows = s.execute(
        text("""
        SELECT
            g.id AS game_id,
            ht.name AS home_team,
            at.name AS away_team,
            g.scheduled_utc,
            g.status,
            p.probability,
            go.bookmaker,
            go.home_price,
            go.away_price,
            go.snapshot
        FROM predictions p
        JOIN games g ON g.id = p.game_id
        JOIN sports sp ON sp.id = g.sport_id
        JOIN teams ht ON ht.id = g.home_team_id
        JOIN teams at ON at.id = g.away_team_id
        LEFT JOIN game_odds go ON go.game_id = g.id AND go.market = 'h2h'
        WHERE sp.code = 'nba'
          AND g.scheduled_utc > NOW() - INTERVAL '24 hours'
          AND p.target = 'home_won'
        ORDER BY g.scheduled_utc, go.bookmaker
    """)
    ).fetchall()

    print(f"\nNBA predictions with odds ({len(rows)} rows):")
    for r in rows:
        print(f"  {r.away_team} @ {r.home_team} | {r.scheduled_utc} | status={r.status}")
        print(
            f"    prob={float(r.probability):.3f} | book={r.bookmaker} | home_ml={r.home_price} away_ml={r.away_price} | snap={r.snapshot}"
        )

    # Now try select_top_picks
    print("\n--- select_top_picks (draftkings) ---")
    picks = select_top_picks(s, "nba", bookmaker="draftkings", n=10)
    print(f"  Qualifying picks: {len(picks)}")
    for p in picks:
        pick_team = p.home_team if p.pick == "home" else p.away_team
        print(
            f"  PICK: {pick_team} | model_prob={p.model_prob:.1%} | edge={p.edge:+.1%} | odds={p.odds_american}"
        )
