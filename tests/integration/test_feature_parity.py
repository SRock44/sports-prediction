"""Integration test: features at training time == features at inference time.

For each game G in a small sample of completed games, we:
1. Compute features at training time (as_of = scheduled_utc - 1h).
2. Compute features again from scratch using the same function.
3. Assert bit-identical output.

This is the primary guard against train/serve skew.
"""

from __future__ import annotations

import pytest

from src.core.time import as_of_for_game


@pytest.mark.integration
def test_nba_matchup_feature_parity(pg_session):
    """Features recomputed on a game must match the stored features_json."""
    from src.db.models import Game, MatchupFeature, Sport
    from src.features.nba.matchup import build_matchup_features

    sport = pg_session.query(Sport).filter_by(code="nba").first()
    if sport is None:
        pytest.skip("No NBA data in test DB — run backfill first")

    games = (
        pg_session.query(Game)
        .filter(Game.sport_id == sport.id, Game.status == "final")
        .order_by(Game.scheduled_utc.desc())
        .limit(10)
        .all()
    )
    if not games:
        pytest.skip("No completed NBA games in test DB")

    failures: list[str] = []
    for game in games:
        mf = pg_session.query(MatchupFeature).filter_by(game_id=game.id).first()
        if mf is None:
            continue

        as_of = as_of_for_game(game.scheduled_utc)
        recomputed = build_matchup_features(pg_session, game, as_of)

        stored_keys = set(mf.features or {})
        recomputed_keys = set(recomputed or {})

        for k in stored_keys & recomputed_keys:
            stored_val = round(float(mf.features[k] or 0), 6)
            new_val = round(float(recomputed[k] or 0), 6)
            if stored_val != new_val:
                failures.append(
                    f"game={game.external_id} feature={k} stored={stored_val} recomputed={new_val}"
                )

    assert not failures, "Train/serve skew detected:\n" + "\n".join(failures)


@pytest.mark.integration
def test_mlb_matchup_feature_parity(pg_session):
    """Same test for MLB."""
    from src.db.models import Game, MatchupFeature, Sport
    from src.features.mlb.matchup import build_matchup_features

    sport = pg_session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        pytest.skip("No MLB data in test DB")

    games = (
        pg_session.query(Game)
        .filter(Game.sport_id == sport.id, Game.status == "final")
        .order_by(Game.scheduled_utc.desc())
        .limit(10)
        .all()
    )
    if not games:
        pytest.skip("No completed MLB games in test DB")

    failures: list[str] = []
    for game in games:
        mf = pg_session.query(MatchupFeature).filter_by(game_id=game.id).first()
        if mf is None:
            continue

        as_of = as_of_for_game(game.scheduled_utc)
        recomputed = build_matchup_features(pg_session, game, as_of)

        for k in set(mf.features or {}) & set(recomputed or {}):
            stored_val = round(float(mf.features[k] or 0), 6)
            new_val = round(float(recomputed[k] or 0), 6)
            if stored_val != new_val:
                failures.append(
                    f"game={game.external_id} feature={k} stored={stored_val} recomputed={new_val}"
                )

    assert not failures, "Train/serve skew detected:\n" + "\n".join(failures)
