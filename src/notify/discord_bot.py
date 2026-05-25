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

            if not picks:
                await interaction.followup.send(
                    "You haven't locked in any picks yet.", ephemeral=True
                )
                return

            lines = []
            pending = won = lost = 0
            for p in picks:
                if p.status == "pending":
                    pending += 1
                elif p.status == "won":
                    won += 1
                elif p.status == "lost":
                    lost += 1
                status_emoji = {"pending": "⏳", "won": "✅", "lost": "❌", "push": "➖"}.get(  # noqa: RUF001
                    p.status, "❓"
                )
                legs_summary = []
                for leg in p.legs or []:
                    team = (
                        leg.get("home_team") if leg.get("pick") == "home" else leg.get("away_team")
                    )
                    odds = leg.get("odds_american", 0)
                    odds_str = f"+{odds}" if odds > 0 else str(odds)
                    legs_summary.append(f"{team} ({odds_str})")
                sport_icon = "🏀" if p.sport_code == "nba" else "⚾"
                ev = p.parlay_ev or 0
                lines.append(
                    f"{status_emoji} {sport_icon} **{' + '.join(legs_summary)}** "
                    f"· {p.n_legs}-leg · EV ${ev:+.0f}/100 · _{p.created_at.strftime('%b %-d')}_"
                )
    except Exception as exc:
        log.error("discord.mypicks_failed", error=str(exc))
        await interaction.followup.send("⚠️ Couldn't load your picks.", ephemeral=True)
        return

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
    name="propparlay",
    description="Build a prop parlay ranked by model confidence — 1, 3, or 5 legs",
)
async def propparlay_cmd(
    interaction: discord.Interaction,
    legs: int = 3,
    sport: str = "both",
) -> None:
    await interaction.response.defer(ephemeral=True)

    from datetime import timedelta
    from math import prod

    from src.core.time import utc_now
    from src.db.models import Game, ModelRecord, Player, Prediction, Sport, Team
    from src.db.session import get_sync_session
    from src.notify.discord_embeds import _american_to_implied

    legs = max(1, min(legs, 5))
    sport = sport.lower()

    try:
        now = utc_now()
        candidates: list[dict] = []

        with get_sync_session() as session:
            sport_filter = []
            if sport in ("nba", "mlb"):
                sport_obj = session.query(Sport).filter_by(code=sport).first()
                if sport_obj:
                    sport_filter.append(Game.sport_id == sport_obj.id)

            rows = (
                session.query(Prediction, Game, Player, Sport)
                .join(Game, Prediction.game_id == Game.id)
                .join(Player, Prediction.player_id == Player.id)
                .join(Sport, Game.sport_id == Sport.id)
                .join(ModelRecord, Prediction.model_id == ModelRecord.id)
                .filter(
                    Prediction.player_id.isnot(None),
                    Prediction.target != "home_won",
                    ModelRecord.active.is_(True),
                    Game.scheduled_utc >= now - timedelta(hours=4),
                    Game.scheduled_utc <= now + timedelta(hours=48),
                    *sport_filter,
                )
                .order_by(Prediction.created_at.desc())
                .all()
            )

            seen = set()
            for pred, game, player, sport_obj in rows:
                key = (player.id, pred.target)
                if key in seen:
                    continue
                seen.add(key)

                q = pred.quantiles or {}
                dk_line = q.get("dk_line")
                dk_over_odds = q.get("dk_over_odds")
                if dk_line is None or dk_over_odds is None:
                    continue

                p_over = float(pred.probability or 0.5)
                p_under = 1.0 - p_over
                # Confidence = how certain the model is (max of both directions)
                confidence = max(p_over, p_under)
                bet_over = p_over >= p_under

                dk_implied = _american_to_implied(float(dk_over_odds))
                edge = (p_over - dk_implied) if bet_over else (p_under - (1.0 - dk_implied))

                # Only take picks where model has positive edge in its chosen direction
                if edge <= 0:
                    continue

                # Get odds for our chosen direction
                if bet_over:
                    odds = float(dk_over_odds)
                else:
                    odds = q.get("dk_under_odds") or (
                        -int(dk_over_odds) if dk_over_odds < 0 else -110
                    )

                home = session.get(Team, game.home_team_id)
                away = session.get(Team, game.away_team_id)
                sport_icon = "🏀" if sport_obj.code == "nba" else "⚾"

                candidates.append(
                    {
                        "player_name": player.full_name or "?",
                        "target": pred.target,
                        "dk_line": dk_line,
                        "direction": "OVER" if bet_over else "UNDER",
                        "p_pick": p_over if bet_over else p_under,
                        "confidence": confidence,
                        "edge": edge,
                        "odds": odds,
                        "model_median": float(pred.value or 0),
                        "game_label": f"{away.abbrev if away else '?'} @ {home.abbrev if home else '?'}",
                        "sport_icon": sport_icon,
                    }
                )

        # Sort by model confidence first, edge as tiebreaker
        candidates.sort(key=lambda x: (x["confidence"], x["edge"]), reverse=True)
        picks = candidates[:legs]

        if not picks:
            await interaction.followup.send(
                "No qualifying prop picks with positive model edge today.", ephemeral=True
            )
            return

        # Compute combined parlay odds and probability
        combined_prob = prod(p["p_pick"] for p in picks)
        # Combined American odds from individual odds
        decimal_odds = prod(
            (p["odds"] / 100 + 1) if p["odds"] > 0 else (100 / abs(p["odds"]) + 1) for p in picks
        )
        combined_american = (
            int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
        )
        ev = combined_prob * (decimal_odds - 1) * 100 - (1 - combined_prob) * 100

        # Build embed
        ev_str = f"+${ev:.0f}" if ev >= 0 else f"-${abs(ev):.0f}"
        ev_color = "🟢" if ev >= 0 else "🔴"
        combined_odds_str = (
            f"+{combined_american}" if combined_american > 0 else str(combined_american)
        )

        lines = [
            f"**{len(picks)}-Leg Prop Parlay**  ·  {combined_odds_str}",
            f"Model combined win prob: **{combined_prob:.1%}**",
            f"{ev_color} EV per $100: **{ev_str}**",
            "",
            "─────────────────────────────",
        ]
        for i, p in enumerate(picks, 1):
            bar_filled = round(p["p_pick"] * 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            odds_str = f"+{int(p['odds'])}" if p["odds"] > 0 else str(int(p["odds"]))
            lines += [
                f"**Leg {i}:** {p['sport_icon']} {p['player_name']} — {p['target']} **{p['direction']}** {p['dk_line']}",
                f"`{bar}` **{p['p_pick']:.0%}** model  ·  Edge **{p['edge']:+.1%}**  ·  {odds_str}",
                f"Median projected: **{p['model_median']:.1f}**  ·  _{p['game_label']}_",
                "",
            ]
        lines.append("─────────────────────────────")
        lines.append("_Sorted by model confidence  ·  Not financial advice_")

        await interaction.followup.send(
            embed=discord.Embed(
                title="🎯 Prop Parlay Builder",
                description="\n".join(lines),
                color=0x9B59B6,
            ),
            ephemeral=True,
        )
    except Exception as exc:
        log.error("discord.propparlay_failed", error=str(exc))
        await interaction.followup.send("⚠️ Error building prop parlay.", ephemeral=True)


@bot.tree.command(
    name="props",
    description="Today's player prop edges — where the model disagrees most with the book",
)
async def props_cmd(
    interaction: discord.Interaction,
    sport: str = "nba",
    min_edge: float = 0.03,
) -> None:
    await interaction.response.defer(ephemeral=True)

    from datetime import timedelta

    from src.core.time import utc_now
    from src.db.models import Game, ModelRecord, Player, Prediction, Sport, Team
    from src.db.session import get_sync_session
    from src.notify.discord_embeds import _american_to_implied, build_props_embed

    sport = sport.lower()
    if sport not in ("nba", "mlb"):
        await interaction.followup.send("Sport must be `nba` or `mlb`.", ephemeral=True)
        return

    try:
        now = utc_now()
        props_out: list[dict] = []

        with get_sync_session() as session:
            sport_obj = session.query(Sport).filter_by(code=sport).first()
            if sport_obj is None:
                await interaction.followup.send("Sport not found in DB.", ephemeral=True)
                return

            # Props predictions for games in next 48h
            rows = (
                session.query(Prediction, Game, Player)
                .join(Game, Prediction.game_id == Game.id)
                .join(Player, Prediction.player_id == Player.id)
                .join(ModelRecord, Prediction.model_id == ModelRecord.id)
                .filter(
                    Game.sport_id == sport_obj.id,
                    Prediction.player_id.isnot(None),
                    Prediction.target != "home_won",
                    ModelRecord.active.is_(True),
                    Game.scheduled_utc >= now - timedelta(hours=4),
                    Game.scheduled_utc <= now + timedelta(hours=48),
                )
                .order_by(Prediction.created_at.desc())
                .all()
            )

            seen = set()
            for pred, game, player in rows:
                key = (player.id, pred.target)
                if key in seen:
                    continue
                seen.add(key)

                quant = pred.quantiles or {}
                dk_line = quant.get("dk_line")
                dk_over_odds = quant.get("dk_over_odds")
                if dk_line is None or dk_over_odds is None:
                    continue

                p_over = float(pred.probability or 0.5)
                dk_implied = _american_to_implied(float(dk_over_odds))
                edge_over = p_over - dk_implied
                edge_under = (1.0 - p_over) - (1.0 - dk_implied)
                best_edge = edge_over if abs(edge_over) >= abs(edge_under) else edge_under

                if abs(best_edge) < min_edge:
                    continue

                home = session.get(Team, game.home_team_id)
                away = session.get(Team, game.away_team_id)
                game_label = f"{away.abbrev if away else '?'} @ {home.abbrev if home else '?'}"

                dk_odds = (
                    dk_over_odds if best_edge > 0 else quant.get("dk_under_odds", dk_over_odds)
                )
                props_out.append(
                    {
                        "player_name": player.full_name or "?",
                        "target": pred.target,
                        "dk_line": dk_line,
                        "p_over": p_over,
                        "dk_over_implied": dk_implied,
                        "edge": best_edge,
                        "model_median": float(pred.value or 0),
                        "game_label": game_label,
                        "dk_odds": dk_odds,
                    }
                )

        props_out.sort(key=lambda x: abs(x["edge"]), reverse=True)
        props_out = props_out[:12]

        embed_dict = build_props_embed(props_out, sport)
        await interaction.followup.send(embed=discord.Embed.from_dict(embed_dict), ephemeral=True)
    except Exception as exc:
        log.error("discord.props_failed", error=str(exc))
        await interaction.followup.send("⚠️ Error loading props.", ephemeral=True)


@bot.tree.command(
    name="propsrecord",
    description="Historical props model performance — win rate and ROI by stat",
)
async def propsrecord_cmd(
    interaction: discord.Interaction,
    sport: str = "both",
    days: int = 30,
) -> None:
    await interaction.response.defer()

    from src.core.time import utc_now
    from src.db.models import Game, ModelRecord, Player, PlayerGameStats, Prediction, Sport
    from src.db.session import get_sync_session
    from src.notify.discord_embeds import _american_to_implied

    sport = sport.lower()
    since = utc_now() - __import__("datetime").timedelta(days=days)

    try:
        with get_sync_session() as session:
            sport_filter = []
            if sport in ("nba", "mlb"):
                sport_obj = session.query(Sport).filter_by(code=sport).first()
                if sport_obj:
                    sport_filter.append(Game.sport_id == sport_obj.id)

            rows = (
                session.query(Prediction, Game, Player, Sport)
                .join(Game, Prediction.game_id == Game.id)
                .join(Player, Prediction.player_id == Player.id)
                .join(Sport, Game.sport_id == Sport.id)
                .join(ModelRecord, Prediction.model_id == ModelRecord.id)
                .filter(
                    Prediction.player_id.isnot(None),
                    Prediction.target != "home_won",
                    ModelRecord.active.is_(True),
                    Game.status == "final",
                    Game.scheduled_utc >= since,
                    *sport_filter,
                )
                .all()
            )

            # Resolve actual stat values from player_game_stats
            from src.features.mlb.player import _safe_float as _sf

            nba_paths = {
                "PTS": ("traditional", "points"),
                "REB": ("traditional", "reboundsTotal"),
                "AST": ("traditional", "assists"),
                "3PM": ("traditional", "threePointersMade"),
            }
            mlb_bat = {
                "H": "hits",
                "HR": "homeRuns",
                "RBI": "rbi",
                "TB": "totalBases",
                "K": "strikeOuts",
            }
            mlb_pit = {"PITCHER_K": "strikeOuts", "PITCHER_ER": "earnedRuns"}

            def _actual(pgs_stats: dict, sport_code: str, target: str) -> float | None:
                if sport_code == "nba":
                    path = nba_paths.get(target)
                    if target == "PRA":
                        t = pgs_stats.get("traditional", {})
                        return sum(
                            _sf(t.get(k)) or 0 for k in ("points", "reboundsTotal", "assists")
                        )
                    if path:
                        return _sf((pgs_stats.get(path[0]) or {}).get(path[1]))
                elif sport_code == "mlb":
                    if target in mlb_bat:
                        return _sf((pgs_stats.get("batting") or {}).get(mlb_bat[target]))
                    if target in mlb_pit:
                        return _sf((pgs_stats.get("pitching") or {}).get(mlb_pit[target]))
                return None

            # Build per-stat record: {stat: {wins, total, profit}}
            from collections import defaultdict

            stat_records: dict = defaultdict(lambda: {"wins": 0, "total": 0, "profit": 0.0})

            for pred, game, player, sport_obj in rows:
                q = pred.quantiles or {}
                dk_line = q.get("dk_line")
                dk_over_odds = q.get("dk_over_odds")
                if dk_line is None or dk_over_odds is None:
                    continue

                p_over = float(pred.probability or 0.5)
                dk_implied = _american_to_implied(float(dk_over_odds))
                edge_over = p_over - dk_implied
                bet_over = edge_over >= 0

                pgs = (
                    session.query(PlayerGameStats)
                    .filter_by(player_id=player.id, game_id=game.id)
                    .first()
                )
                if pgs is None:
                    continue

                actual = _actual(pgs.stats or {}, sport_obj.code, pred.target)
                if actual is None:
                    continue

                won = (actual > dk_line) if bet_over else (actual < dk_line)
                odds = float(dk_over_odds) if bet_over else (q.get("dk_under_odds") or -110)
                payout = (odds / 100.0) if odds > 0 else (100.0 / abs(odds))
                profit = payout if won else -1.0

                key = f"{sport_obj.code.upper()} {pred.target}"
                stat_records[key]["total"] += 1
                stat_records[key]["wins"] += int(won)
                stat_records[key]["profit"] += profit

        if not stat_records:
            await interaction.followup.send(
                embed=discord.Embed(
                    title="📊 Props Record",
                    description=f"No settled props in the last {days} days.",
                    color=0x95A5A6,
                )
            )
            return

        lines = [f"_Last {days} days  ·  settled bets only_", ""]
        total_w = total_t = 0
        total_profit = 0.0
        for stat_key in sorted(stat_records):
            r = stat_records[stat_key]
            w, t, profit = r["wins"], r["total"], r["profit"]
            if t == 0:
                continue
            roi = profit / t * 100
            pct = w / t * 100
            bar = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            lines.append(f"{bar} **{stat_key}** — {w}/{t} ({pct:.0f}%)  ROI: {roi:+.1f}%")
            total_w += w
            total_t += t
            total_profit += profit

        if total_t:
            overall_roi = total_profit / total_t * 100
            lines += [
                "",
                f"**Overall: {total_w}/{total_t} ({total_w / total_t * 100:.0f}%)  ROI: {overall_roi:+.1f}%**",
            ]

        await interaction.followup.send(
            embed=discord.Embed(
                title="📊 Props Model Record",
                description="\n".join(lines),
                color=0x2ECC71 if total_profit > 0 else 0xE74C3C,
                footer=discord.EmbedFooter(text="$100 flat bet per pick  ·  Not financial advice"),
            )
        )
    except Exception as exc:
        log.error("discord.propsrecord_failed", error=str(exc))
        await interaction.followup.send("⚠️ Error loading props record.", ephemeral=True)


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
                emoji = "🏀" if sport_code == "nba" else "⚾"

                # ── Winner model ──────────────────────────────────────────────
                champ = (
                    session.query(ModelRecord)
                    .filter_by(sport_id=sport.id, kind="winner", active=True)
                    .first()
                )
                if champ is None:
                    lines.append(f"**{sport_code.upper()} Winner:** No active model")
                else:
                    m = champ.metrics or {}
                    lines.append(
                        f"{emoji} **{sport_code.upper()} Winner** — v`{champ.version[:8]}`  "
                        f"· id={champ.id}"
                    )
                    lines.append(
                        f"  Log-loss: `{m.get('logloss', '?'):.4f}`  "
                        f"Acc: `{m.get('accuracy', 0):.1%}`  "
                        f"ECE: `{m.get('ece', '?'):.4f}`  "
                        f"· _{champ.trained_at.strftime('%Y-%m-%d') if champ.trained_at else '?'}_"
                    )

                # ── Props models ──────────────────────────────────────────────
                props_models = (
                    session.query(ModelRecord)
                    .filter_by(sport_id=sport.id, kind="props", active=True)
                    .order_by(ModelRecord.target)
                    .all()
                )
                if props_models:
                    lines.append(f"  **Props** ({len(props_models)} stats):")
                    for pm in props_models:
                        mae = (pm.metrics or {}).get("mae_median")
                        cov = (pm.metrics or {}).get("coverage_80")
                        mae_str = f"`{mae:.3f}`" if mae is not None else "`?`"
                        cov_str = f"{cov:.0%}" if cov is not None else "?"
                        lines.append(
                            f"    id={pm.id} `{pm.target:<10}` MAE {mae_str}"
                            f"  cov80 {cov_str}  · v`{pm.version[:8]}`"
                        )
                else:
                    lines.append("  **Props:** No active models")

                lines.append("")

        embed = discord.Embed(
            title="🤖 Model Info",
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
