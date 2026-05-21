"""Telegram notifier using the Bot API (no python-telegram-bot dep — plain httpx).

We use MarkdownV2 formatting. All special chars must be escaped.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"

_ESCAPE_CHARS = r"\_*[]()~`>#+-=|{}.!"


def _esc(text: str) -> str:
    """Escape all MarkdownV2 special characters."""
    for ch in _ESCAPE_CHARS:
        text = text.replace(ch, f"\\{ch}")
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15), reraise=False)
def send_game_prediction(
    bot_token: str,
    chat_id: str | int,
    game_info: dict[str, Any],
    prediction: dict[str, Any],
    props: list[dict[str, Any]] | None = None,
    is_lineup_update: bool = False,
) -> bool:
    """Send a MarkdownV2-formatted game prediction to a Telegram chat."""
    home_prob = float(prediction.get("home_win_probability", 0.5))
    away_prob = round(1.0 - home_prob, 3)
    home_team = game_info.get("home_team", {}).get("name", "Home")
    away_team = game_info.get("away_team", {}).get("name", "Away")
    scheduled = game_info.get("scheduled_utc", "")[:16].replace("T", " ")
    bar = _prob_bar(home_prob)

    prefix = "📋 *LINEUP UPDATE*\n" if is_lineup_update else "🎯 "
    title = f"{prefix}*{_esc(away_team)} @ {_esc(home_team)}*"

    lines = [
        title,
        f"_{_esc(scheduled)} UTC_",
        "",
        "*Win Probability*",
        f"`{bar}`",
        f"*{_esc(home_team)}* {home_prob:.1%}  ·  *{_esc(away_team)}* {away_prob:.1%}",
    ]

    if props:
        lines += ["", "*Top Props*"]
        for prop in props[:3]:
            player = _esc(prop.get("player_name", "?"))
            target = _esc(prop.get("target", "?"))
            median = prop.get("predicted_median", 0)
            qs = prop.get("quantiles") or {}
            lo = qs.get("0.1", median - 2)
            hi = qs.get("0.9", median + 2)
            lines.append(
                f"• {player} {target}: *{_esc(f'{median:.1f}')}* _{_esc(f'{lo:.1f}–{hi:.1f}')}_"
            )

    mv = _esc(prediction.get("model_version", "?"))
    ao = _esc(prediction.get("as_of_utc", "")[:16])
    lines.append(f"\n_model v{mv} · {ao} UTC_")

    text = "\n".join(lines)

    url = _TG_API.format(token=bot_token)
    resp = httpx.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        },
        timeout=10,
    )
    resp.raise_for_status()
    log.info("telegram.sent", chat_id=chat_id, home=home_team, away=away_team)
    return True


def send_ops_alert(bot_token: str, chat_id: str | int, message: str) -> None:
    try:
        url = _TG_API.format(token=bot_token)
        httpx.post(
            url,
            json={
                "chat_id": chat_id,
                "text": _esc(message),
                "parse_mode": "MarkdownV2",
            },
            timeout=5,
        )
    except Exception as exc:
        log.warning("telegram.ops_alert_failed", error=str(exc))


def _prob_bar(p: float, width: int = 10) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled) + f" {p:.0%}"
