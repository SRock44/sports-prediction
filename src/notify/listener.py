"""PostgreSQL LISTEN/NOTIFY-based notifier process.

Entry point for the `notifier` Docker service.
Listens on `predictions_channel`; for each payload dispatches to Discord
and/or Telegram after dedup check.

Payload emitted by the score.py pg_notify call:
  {"game_id": <int>, "target": "home_win", "probability": 0.63, "is_lineup_update": false}
"""

from __future__ import annotations

import json
import select
import signal
import sys
import time
from typing import Any

import psycopg2
import psycopg2.extensions
import redis

from src.core.config import get_settings
from src.core.logging import get_logger
from src.db.models import Game, ModelRecord, Prediction, Sport, Team
from src.db.session import get_sync_session
from src.notify import discord, telegram
from src.notify.dedup import record_send, should_send

log = get_logger(__name__)

_CHANNEL = "predictions_channel"
_RECONNECT_DELAY = 5  # seconds between reconnect attempts


def _build_game_info(session: Any, game: Game) -> dict[str, Any]:
    home_team = session.get(Team, game.home_team_id)
    away_team = session.get(Team, game.away_team_id)
    return {
        "id": game.id,
        "external_id": game.external_id,
        "scheduled_utc": game.scheduled_utc.isoformat() if game.scheduled_utc else "",
        "home_team": {"name": home_team.name if home_team else "Home"},
        "away_team": {"name": away_team.name if away_team else "Away"},
    }


def _build_prediction_payload(
    session: Any,
    game_id: int,
    target: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (winner_prediction, props_list) for the notification."""
    winner = (
        session.query(Prediction)
        .join(ModelRecord)
        .filter(
            Prediction.game_id == game_id,
            Prediction.target == target,
            ModelRecord.active.is_(True),
        )
        .order_by(Prediction.created_at.desc())
        .first()
    )
    if winner is None:
        return None, []

    model = session.get(ModelRecord, winner.model_id)
    pred_dict = {
        "home_win_probability": float(winner.probability or 0.5),
        "model_version": model.version if model else "?",
        "as_of_utc": winner.created_at.isoformat() if winner.created_at else "",
    }

    props_rows = (
        session.query(Prediction)
        .join(ModelRecord)
        .filter(
            Prediction.game_id == game_id,
            Prediction.target != "home_win",
            Prediction.player_id.isnot(None),
            ModelRecord.active.is_(True),
        )
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )
    props = []
    for p in props_rows:
        props.append(
            {
                "player_name": p.player.full_name if p.player else "?",
                "target": p.target,
                "predicted_median": float(p.value or 0),
                "quantiles": p.quantiles or {},
            }
        )

    return pred_dict, props


def _dispatch(payload: dict[str, Any], r: redis.Redis, settings: Any) -> None:
    game_id = payload.get("game_id")
    target = payload.get("target", "home_win")
    probability = float(payload.get("probability", 0.5))
    is_lineup_update = bool(payload.get("is_lineup_update", False))

    if not should_send(r, game_id, target, probability, is_lineup_update):
        return

    with get_sync_session() as session:
        game = session.get(Game, game_id)
        if game is None:
            log.warning("notify.game_not_found", game_id=game_id)
            return

        game_info = _build_game_info(session, game)
        prediction, props = _build_prediction_payload(session, game_id, target)
        if prediction is None:
            log.warning("notify.prediction_not_found", game_id=game_id, target=target)
            return

        sport = session.get(Sport, game.sport_id)
        sport_code = sport.code if sport else "nba"

    # Attempt delivery to all channels; track whether at least one succeeded
    # so we only commit the dedup record after a confirmed send.
    delivered = False

    # Discord webhooks
    webhook_urls: list[str] = []
    if sport_code == "nba" and settings.discord_webhook_nba:
        webhook_urls.append(settings.discord_webhook_nba)
    if sport_code == "mlb" and settings.discord_webhook_mlb:
        webhook_urls.append(settings.discord_webhook_mlb)

    for url in webhook_urls:
        try:
            discord.send_game_prediction(url, game_info, prediction, props, is_lineup_update)
            delivered = True
        except Exception as exc:
            log.warning("notify.discord_failed", error=str(exc))

    # Telegram chats
    if settings.telegram_bot_token:
        chat_ids: list[str] = []
        if sport_code == "nba" and settings.telegram_chat_id_nba:
            chat_ids.append(settings.telegram_chat_id_nba)
        if sport_code == "mlb" and settings.telegram_chat_id_mlb:
            chat_ids.append(settings.telegram_chat_id_mlb)

        for chat_id in chat_ids:
            try:
                telegram.send_game_prediction(
                    settings.telegram_bot_token,
                    chat_id,
                    game_info,
                    prediction,
                    props,
                    is_lineup_update,
                )
                delivered = True
            except Exception as exc:
                log.warning("notify.telegram_failed", error=str(exc))

    if delivered:
        record_send(r, game_id, target, probability)


def _listen_loop(conn: psycopg2.extensions.connection, r: redis.Redis, settings: Any) -> None:
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute(f"LISTEN {_CHANNEL};")
    log.info("notify.listening", channel=_CHANNEL)

    while True:
        if select.select([conn], [], [], 30)[0]:
            conn.poll()
            while conn.notifies:
                note = conn.notifies.pop(0)
                try:
                    payload = json.loads(note.payload)
                    log.info("notify.received", payload=payload)
                    _dispatch(payload, r, settings)
                except Exception as exc:
                    log.exception("notify.dispatch_error", error=str(exc))


def run() -> None:
    settings = get_settings()
    r = redis.from_url(settings.redis_url, decode_responses=True)

    _shutdown = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal _shutdown
        log.info("notify.shutdown_signal", signal=signum)
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not _shutdown:
        try:
            conn = psycopg2.connect(settings.database_url_sync)
            _listen_loop(conn, r, settings)
        except Exception as exc:
            log.error("notify.connection_error", error=str(exc))
            if not _shutdown:
                time.sleep(_RECONNECT_DELAY)
        finally:
            from contextlib import suppress

            with suppress(Exception):
                conn.close()  # type: ignore[possibly-undefined]

    log.info("notify.stopped")
    sys.exit(0)


if __name__ == "__main__":
    run()
