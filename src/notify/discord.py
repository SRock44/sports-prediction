"""Discord webhook notifier with rich embeds."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

log = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15), reraise=False)
def send_game_prediction(
    webhook_url: str,
    game_info: dict[str, Any],
    prediction: dict[str, Any],
    props: list[dict[str, Any]] | None = None,
    is_lineup_update: bool = False,
) -> bool:
    """Send a rich embed to a Discord webhook. Returns True on success."""
    home_prob = float(prediction.get("home_win_probability", 0.5))
    away_prob = round(1.0 - home_prob, 3)
    home_team = game_info.get("home_team", {}).get("name", "Home")
    away_team = game_info.get("away_team", {}).get("name", "Away")
    scheduled = game_info.get("scheduled_utc", "")[:16].replace("T", " ")

    bar = _prob_bar(home_prob)

    title_prefix = "📋 LINEUP UPDATE — " if is_lineup_update else "🎯 "
    title = f"{title_prefix}{away_team} @ {home_team}"

    description_lines = [
        f"**{scheduled} UTC**",
        "",
        "**Win Probability**",
        f"`{bar}`",
        f"**{home_team}** {home_prob:.1%} · **{away_team}** {away_prob:.1%}",
    ]

    if props:
        description_lines += ["", "**Top Props**"]
        for prop in props[:3]:
            player = prop.get("player_name", "?")
            target = prop.get("target", "?")
            median = prop.get("predicted_median", 0)
            qs = prop.get("quantiles") or {}
            lo = qs.get("0.1", median - 2)
            hi = qs.get("0.9", median + 2)
            description_lines.append(f"• {player} {target}: **{median:.1f}** _{lo:.1f}-{hi:.1f}_")

    embed = {
        "title": title,
        "description": "\n".join(description_lines),
        "color": 0x1D82B6,  # NBA blue
        "footer": {
            "text": f"model v{prediction.get('model_version', '?')} · {prediction.get('as_of_utc', '')[:16]} UTC"
        },
    }

    payload = {"embeds": [embed]}

    resp = httpx.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()
    log.info("discord.sent", game=title)
    return True


def _prob_bar(p: float, width: int = 10) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled) + f" {p:.0%}"


def send_ops_alert(webhook_url: str, message: str) -> None:
    try:
        httpx.post(webhook_url, json={"content": message}, timeout=5)
    except Exception as exc:
        log.warning("discord.ops_alert_failed", error=str(exc))
