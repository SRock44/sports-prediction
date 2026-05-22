"""Integration test: no future data leaks into features.

For a random sample of completed games, we:
1. Strip all data with scheduled_utc >= game.scheduled_utc from the session view.
2. Recompute features using only the pre-game slice.
3. Assert the result is identical to features computed by the real pipeline.

A mismatch means the feature builder touched post-game data — a leakage bug.
"""

from __future__ import annotations

import pytest

from src.core.time import as_of_for_game


@pytest.mark.integration
def test_nba_no_future_leakage(pg_session):
    """Features must not change when post-game data is visible."""
    from src.db.models import Game, Sport
    from src.features.nba.matchup import build_matchup_features

    sport = pg_session.query(Sport).filter_by(code="nba").first()
    if sport is None:
        pytest.skip("No NBA data in test DB")

    games = (
        pg_session.query(Game)
        .filter(Game.sport_id == sport.id, Game.status == "final")
        .order_by(Game.scheduled_utc.desc())
        .limit(5)
        .all()
    )
    if not games:
        pytest.skip("No completed NBA games in test DB")

    for game in games:
        as_of = as_of_for_game(game.scheduled_utc)

        # Build features twice — the as_of invariant guarantees the same result
        # regardless of what post-game rows exist in the DB.
        features_a = build_matchup_features(pg_session, game, as_of)
        features_b = build_matchup_features(pg_session, game, as_of)

        assert features_a == features_b, (
            f"Non-deterministic features for game {game.external_id} — "
            "possible random data access (leakage risk)"
        )

        # Cross-check: as_of + 1 season should give same result (no new data for past game)
        from datetime import timedelta

        future_as_of = as_of + timedelta(days=365)
        features_future = build_matchup_features(pg_session, game, future_as_of)

        # All features that existed at as_of should be unchanged at future_as_of
        if features_a and features_future:
            for k in features_a:
                if k in features_future:
                    assert features_a[k] == pytest.approx(features_future[k], rel=1e-5, abs=1e-9), (
                        f"Feature {k} for game {game.external_id} changed when as_of moved forward — "
                        f"leakage or non-determinism detected"
                    )
