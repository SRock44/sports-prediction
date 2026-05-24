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
from src.notify.discord_views import ConfirmParlayView, KirkovaView, SportSelectView

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
        # Register ConfirmParlayView as persistent so "Lock It In" buttons on old
        # /predict embeds keep working after bot restarts. KirkovaView uses a dynamic
        # select menu (can't be persistent), so /kirkova embeds need a re-run after restart.
        self.add_view(ConfirmParlayView(parlay=None, sport=""))

        if self._guild:
            self.tree.copy_global_to(guild=self._guild)
            synced = await self.tree.sync(guild=self._guild)
            log.info("discord_bot.commands_synced", guild=self._guild.id, count=len(synced))
        else:
            synced = await self.tree.sync()
            log.info("discord_bot.global_sync", count=len(synced))

    async def on_ready(self) -> None:
        log.info("discord_bot.ready", user=str(self.user))
        if self.user:
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
        from src.notify.discord_views import _locked_game_ids

        already_locked = _locked_game_ids(str(interaction.user.id))

        with get_sync_session() as session:
            nba_all = _fetch_candidate_legs(session, "nba", "draftkings")
            mlb_all = _fetch_candidate_legs(session, "mlb", "draftkings")

        nba_legs = [leg for leg in nba_all if leg.game_id not in already_locked]
        mlb_legs = [leg for leg in mlb_all if leg.game_id not in already_locked]
        all_legs = nba_legs + mlb_legs
        total_qualifying = len(nba_all) + len(mlb_all)
    except Exception as exc:
        log.error("discord.kirkova_failed", error=str(exc))
        await interaction.followup.send("⚠️ Error loading predictions.", ephemeral=True)
        return

    if not all_legs:
        if total_qualifying > 0:
            # User locked everything already
            embed = discord.Embed(
                title="✅ You've locked all available picks",
                description=(
                    f"You've already tracked every qualifying pick today ({total_qualifying} total).\n"
                    "Use `/mypicks` to see your picks, or check back tomorrow."
                ),
                color=0x2ECC71,
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
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
    name="mypicks",
    description="Show all your pending tracked picks",
)
async def mypicks(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)

    from src.db.models.discord_parlay import DiscordParlay
    from src.db.session import get_sync_session

    try:
        with get_sync_session() as session:
            picks = (
                session.query(DiscordParlay)
                .filter_by(discord_user_id=str(interaction.user.id))
                .order_by(DiscordParlay.created_at.desc())
                .limit(20)
                .all()
            )
    except Exception as exc:
        log.error("discord.mypicks_failed", error=str(exc))
        await interaction.followup.send("⚠️ Couldn't load your picks.", ephemeral=True)
        return

    if not picks:
        await interaction.followup.send("You haven't locked in any picks yet.", ephemeral=True)
        return

    lines = []
    for p in picks:
        status_emoji = {"pending": "⏳", "won": "✅", "lost": "❌", "push": "➖"}.get(  # noqa: RUF001
            p.status, "❓"
        )
        legs_summary = []
        for leg in p.legs or []:
            team = leg.get("home_team") if leg.get("pick") == "home" else leg.get("away_team")
            odds = leg.get("odds_american", 0)
            odds_str = f"+{odds}" if odds > 0 else str(odds)
            legs_summary.append(f"{team} ({odds_str})")
        sport_icon = "🏀" if p.sport_code == "nba" else "⚾"
        lines.append(
            f"{status_emoji} {sport_icon} **{' + '.join(legs_summary)}** "
            f"· {p.n_legs}-leg · EV ${p.parlay_ev:+.0f}/100 · _{p.created_at.strftime('%b %-d')}_"
        )

    pending = sum(1 for p in picks if p.status == "pending")
    won = sum(1 for p in picks if p.status == "won")
    lost = sum(1 for p in picks if p.status == "lost")

    embed = discord.Embed(
        title=f"📋 {interaction.user.display_name}'s Picks",
        description="\n".join(lines),
        color=0x1D82B6,
    )
    embed.set_footer(
        text=f"⏳ {pending} pending  ·  ✅ {won} won  ·  ❌ {lost} lost  ·  Last 20 shown"
    )
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(
    name="training",
    description="Show recent model training runs",
)
async def training_log(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)

    import os

    log_path = "/app/reports/training_log.md"
    if not os.path.exists(log_path):
        await interaction.followup.send("No training log yet — runs tonight.", ephemeral=True)
        return

    with open(log_path) as f:
        content = f.read()

    # Show last ~1800 chars (last few runs) to fit in embed
    if len(content) > 1800:
        content = "…" + content[-1800:]

    embed = discord.Embed(
        title="🤖 Training Log",
        description=f"```\n{content}\n```",
        color=0x9B59B6,
    )
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(
    name="enzo",
    description="Objective model predictions for every game — who wins and why (no odds/betting)",
)
async def enzo(interaction: discord.Interaction) -> None:
    await interaction.response.defer()

    from src.db.session import get_sync_session
    from src.notify.enzo import build_enzo_embeds, fetch_enzo_games

    try:
        with get_sync_session() as session:
            games = fetch_enzo_games(session)
        embeds = build_enzo_embeds(games)
    except Exception as exc:
        log.error("discord.enzo_failed", error=str(exc))
        await interaction.followup.send("⚠️ Error loading predictions.", ephemeral=True)
        return

    # Send first embed as the initial followup, rest as additional messages
    first, *rest = embeds
    await interaction.followup.send(embed=discord.Embed.from_dict(first))
    for embed_dict in rest:
        await interaction.followup.send(embed=discord.Embed.from_dict(embed_dict))


@bot.tree.command(
    name="record",
    description="Show today's model prediction win/loss record",
)
async def record(interaction: discord.Interaction) -> None:
    await interaction.response.defer()

    from src.tasks.outcome_tasks import _get_season_record
    from src.tasks.train_tasks import _get_calibration_stats

    nba_rec = _get_season_record("nba")
    mlb_rec = _get_season_record("mlb")
    nba_cal = _get_calibration_stats("nba")
    mlb_cal = _get_calibration_stats("mlb")

    embed = discord.Embed(
        title="📊 Model Season Record",
        description="\n".join(
            [
                f"🏀 **NBA:** {nba_rec}",
                f"  Log-loss: `{nba_cal}`",
                f"⚾ **MLB:** {mlb_rec}",
                f"  Log-loss: `{mlb_cal}`",
                "",
                "_Last 180 days, active model only_",
                "_Log-loss < 0.69 = better than random_",
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
