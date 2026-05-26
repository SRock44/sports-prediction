"""Celery tasks: rebuild matchup and team features for upcoming games."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from celery import shared_task
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import as_of_for_game, utc_now
from src.db.models import Game, MatchupFeature, Sport
from src.db.session import get_sync_session
from src.ingest.common import dict_hash

log = get_logger(__name__)


@shared_task(name="src.tasks.feature_tasks.rebuild_features_nba", bind=True, max_retries=3)
def rebuild_features_nba(self: Any) -> dict[str, Any]:
    return _rebuild_features("nba")


@shared_task(name="src.tasks.feature_tasks.rebuild_features_mlb", bind=True, max_retries=3)
def rebuild_features_mlb(self: Any) -> dict[str, Any]:
    return _rebuild_features("mlb")


def _rebuild_features(sport_code: str) -> dict[str, Any]:
    """Rebuild matchup features for all upcoming games (next 48 h)."""
    built = 0
    errors = 0

    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code=sport_code).first()
        if sport is None:
            log.warning("feature_tasks.no_sport", sport=sport_code)
            return {"sport": sport_code, "built": 0, "errors": 0}

        now = utc_now()
        window_end = now + timedelta(hours=48)

        upcoming: list[Game] = (
            session.query(Game)
            .filter(
                Game.sport_id == sport.id,
                Game.scheduled_utc >= now,
                Game.scheduled_utc <= window_end,
                Game.status.notin_(["final", "cancelled", "postponed"]),
            )
            .all()
        )

        log.info(
            "feature_tasks.start",
            sport=sport_code,
            games=len(upcoming),
        )

        for game in upcoming:
            try:
                as_of = as_of_for_game(game.scheduled_utc)
                features = _compute_matchup_features(session, sport_code, game, as_of)
                if features is None:
                    continue

                fhash = dict_hash(features)
                existing = session.query(MatchupFeature).filter_by(game_id=game.id).first()
                if existing is None:
                    session.add(
                        MatchupFeature(
                            game_id=game.id,
                            features=features,
                            computed_at=utc_now(),
                        )
                    )
                else:
                    existing.features = features
                    existing.computed_at = utc_now()

                session.flush()
                built += 1
                log.debug(
                    "feature_tasks.built",
                    sport=sport_code,
                    game_id=game.external_id,
                    features_hash=fhash[:8],
                )
            except Exception as exc:
                errors += 1
                log.exception(
                    "feature_tasks.error",
                    sport=sport_code,
                    game_id=game.external_id,
                    error=str(exc),
                )

        session.commit()

    log.info("feature_tasks.done", sport=sport_code, built=built, errors=errors)
    return {"sport": sport_code, "built": built, "errors": errors}


@shared_task(name="src.tasks.feature_tasks.backfill_features_mlb", bind=True, max_retries=1)
def backfill_features_mlb(self: Any, season_from: int = 2022) -> dict[str, Any]:
    """Recompute matchup_features for all historical final MLB games.

    Used after adding new features to ensure the full training set has real values
    instead of 0.0 fill-value defaults. One-off task; not in beat schedule.
    """
    built = 0
    errors = 0

    with get_sync_session() as session:
        from sqlalchemy import text

        sport = session.query(Sport).filter_by(code="mlb").first()
        if sport is None:
            return {"built": 0, "errors": 0}

        rows = session.execute(
            text("""
                SELECT g.id, g.external_id, g.scheduled_utc
                FROM games g
                JOIN matchup_features mf ON mf.game_id = g.id
                WHERE g.sport_id = :sid
                  AND g.status = 'final'
                  AND g.home_score IS NOT NULL
                  AND g.season >= :season_from
                ORDER BY g.scheduled_utc
            """),
            {"sid": sport.id, "season_from": season_from},
        ).fetchall()

        log.info("backfill_features.start", sport="mlb", total=len(rows))

        for _i, row in enumerate(rows):
            game = session.query(Game).get(row.id)
            if game is None:
                continue
            try:
                as_of = as_of_for_game(row.scheduled_utc)
                features = _compute_matchup_features(session, "mlb", game, as_of)
                if features is None:
                    errors += 1
                    continue

                existing = session.query(MatchupFeature).filter_by(game_id=game.id).first()
                if existing:
                    existing.features = features
                    existing.computed_at = utc_now()
                else:
                    session.add(
                        MatchupFeature(game_id=game.id, features=features, computed_at=utc_now())
                    )

                built += 1
                if built % 200 == 0:
                    session.commit()
                    log.info(
                        "backfill_features.progress", built=built, errors=errors, total=len(rows)
                    )
            except Exception as exc:
                errors += 1
                log.warning("backfill_features.error", game_id=row.id, error=str(exc))

        session.commit()

    log.info("backfill_features.done", sport="mlb", built=built, errors=errors)
    return {"built": built, "errors": errors}


def _compute_matchup_features(
    session: Session,
    sport_code: str,
    game: Game,
    as_of: Any,
) -> dict[str, Any] | None:
    """Dispatch to sport-specific feature builder."""
    try:
        if sport_code == "nba":
            from src.features.nba.matchup import build_matchup_features

            return build_matchup_features(session, game, as_of)
        elif sport_code == "mlb":
            from src.features.mlb.matchup import build_matchup_features

            return build_matchup_features(session, game, as_of)
        else:
            log.warning("feature_tasks.unknown_sport", sport=sport_code)
            return None
    except Exception as exc:
        log.exception(
            "feature_tasks.build_failed",
            sport=sport_code,
            game_id=getattr(game, "external_id", "?"),
            error=str(exc),
        )
        return None
