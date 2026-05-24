"""Interactive Discord bot — slash commands + button UI for predictions.

Entry point: python -m src.notify.discord_bot
Docker service: discord-bot

Flow:
  /predict → SportSelect → BookSelect → ParlaySize (1/3/5) → Embed + ConfirmView

Proactive posting (via Celery, not the bot):
  post_daily_picks (14:00 UTC) → 10 best picks posted via webhook
  check_outcomes (every 5 min)  → wins posted immediately via webhook
"""

from __future__ import annotations

import signal
import sys

import discord
from discord import app_commands

from src.core.config import settings
from src.core.logging import get_logger
from src.notify.discord_views import KirkovaView, SportSelectView

log = get_logger(__name__)


# ── Bot setup ─────────────────────────────────────────────────────────────────


class PredictionBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._guild: discord.Object | None = (
            discord.Object(id=int(settings.discord_guild_id)) if settings.discord_guild_id else None
        )

    async def setup_hook(self) -> None:
        if self._guild:
            # Sync to guild instantly (dev/production with a known guild ID)
            self.tree.copy_global_to(guild=self._guild)
            synced = await self.tree.sync(guild=self._guild)
            log.info("discord_bot.commands_synced", guild=self._guild.id, count=len(synced))
        else:
            # Global sync — can take up to 1 hour to propagate
            synced = await self.tree.sync()
            log.info("discord_bot.global_sync", count=len(synced))

    async def on_ready(self) -> None:
        log.info("discord_bot.ready", user=str(self.user))
        print(f"[Bot] Logged in as {self.user} (id={self.user.id})")


bot = PredictionBot()


# ── Slash commands ────────────────────────────────────────────────────────────


@bot.tree.command(
    name="predict",
    description="Get AI-powered game predictions and build a parlay",
)
async def predict(interaction: discord.Interaction) -> None:
    """Trigger the parlay builder flow. Only visible to the requesting user until posted."""
    view = SportSelectView()
    await interaction.response.send_message(
        "🎯 **Build your parlay** — pick a sport to start:",
        view=view,
        ephemeral=True,  # Private until the user confirms and it gets posted
    )


@bot.tree.command(
    name="kirkova",
    description="See ALL of today's predictions — pick the ones you want to lock in",
)
async def kirkova(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)

    from src.db.session import get_sync_session
    from src.notify.discord_embeds import build_kirkova_embed

    try:
        from src.models.parlay import _fetch_candidate_legs

        with get_sync_session() as session:
            nba_legs = _fetch_candidate_legs(session, "nba", "draftkings")
            mlb_legs = _fetch_candidate_legs(session, "mlb", "draftkings")
        all_legs = nba_legs + mlb_legs
    except Exception as exc:
        log.error("discord.kirkova_failed", error=str(exc))
        await interaction.followup.send("⚠️ Error loading predictions.", ephemeral=True)
        return

    if not all_legs:
        await interaction.followup.send(
            "No high-confidence picks right now. Check back once more games are priced.",
            ephemeral=True,
        )
        return

    embed_dict = build_kirkova_embed(nba_legs, mlb_legs)
    view = KirkovaView(all_legs)
    await interaction.followup.send(
        content="🎯 Today's value picks — select the ones you want, then lock 'em in.",
        embed=discord.Embed.from_dict(embed_dict),
        view=view,
    )


@bot.tree.command(
    name="record",
    description="Show today's model prediction win/loss record",
)
async def record(interaction: discord.Interaction) -> None:
    await interaction.response.defer()

    from src.tasks.outcome_tasks import _get_season_record

    nba_rec = _get_season_record("nba")
    mlb_rec = _get_season_record("mlb")

    embed = discord.Embed(
        title="📊 Model Season Record",
        description="\n".join(
            [
                f"🏀 **NBA:** {nba_rec}",
                f"⚾ **MLB:** {mlb_rec}",
                "",
                "_Last 180 days, active model only_",
            ]
        ),
        color=0x1D82B6,
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(
    name="model",
    description="Show current champion model stats",
)
async def model_info(interaction: discord.Interaction) -> None:
    await interaction.response.defer()

    from src.db.models import ModelRecord, Sport
    from src.db.session import get_sync_session

    try:
        with get_sync_session() as session:
            lines = []
            for sport_code in ("nba", "mlb"):
                sport = session.query(Sport).filter_by(code=sport_code).first()
                if sport is None:
                    continue
                champ = (
                    session.query(ModelRecord)
                    .filter_by(sport_id=sport.id, kind="winner", active=True)
                    .first()
                )
                if champ is None:
                    lines.append(f"**{sport_code.upper()}:** No active model")
                    continue
                m = champ.metrics or {}
                emoji = "🏀" if sport_code == "nba" else "⚾"
                lines.append(
                    f"{emoji} **{sport_code.upper()}** (id={champ.id}  v{champ.version[:8]})"
                )
                lines.append(
                    f"  Log-loss: `{m.get('logloss', '?'):.4f}`  "
                    f"Accuracy: `{m.get('accuracy', 0):.1%}`  "
                    f"ECE: `{m.get('ece', '?'):.4f}`"
                )
                lines.append(
                    f"  Trained: {champ.trained_at.strftime('%Y-%m-%d') if champ.trained_at else '?'}"
                )
                lines.append("")

        embed = discord.Embed(
            title="🤖 Champion Model Info",
            description="\n".join(lines) or "No active models found.",
            color=0x9B59B6,
        )
        await interaction.followup.send(embed=embed)
    except Exception as exc:
        log.error("discord.model_info_failed", error=str(exc))
        await interaction.followup.send("⚠️ Could not load model info.", ephemeral=True)


# ── Entry point ───────────────────────────────────────────────────────────────


def _handle_shutdown(sig: int, frame: object) -> None:
    log.info("discord_bot.shutdown", signal=sig)
    sys.exit(0)


def main() -> None:
    token = settings.discord_bot_token
    if not token:
        print("[Bot] ERROR: DISCORD_BOT_TOKEN not set in .env — cannot start.", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    log.info("discord_bot.starting")
    bot.run(token.get_secret_value(), log_handler=None)


if __name__ == "__main__":
    main()
