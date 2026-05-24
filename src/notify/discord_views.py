"""Discord UI components — Views, Selects, Buttons for /predict and /kirkova."""

from __future__ import annotations

import discord

from src.core.logging import get_logger
from src.models.parlay import ParlayLeg

log = get_logger(__name__)


# ── /predict flow ─────────────────────────────────────────────────────────────


class SportSelectView(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=120)

    @discord.ui.select(
        placeholder="Choose a sport…",
        options=[
            discord.SelectOption(label="NBA 🏀", value="nba", description="NBA basketball"),
            discord.SelectOption(label="MLB ⚾", value="mlb", description="MLB baseball"),
        ],
        custom_id="sport_select",
    )
    async def sport_chosen(
        self, interaction: discord.Interaction, select: discord.ui.Select
    ) -> None:
        sport = select.values[0]
        view = ParlayTypeView(sport=sport)
        await interaction.response.edit_message(
            content=f"**{sport.upper()}** selected. Pick parlay size:",
            view=view,
        )


class ParlayTypeView(discord.ui.View):
    def __init__(self, sport: str) -> None:
        super().__init__(timeout=120)
        self.sport = sport

    async def _generate(self, interaction: discord.Interaction, n_legs: int) -> None:
        await interaction.response.defer(ephemeral=True)

        from src.db.session import get_sync_session
        from src.models.parlay import build_parlay
        from src.notify.discord_embeds import (
            build_no_picks_embed,
            build_parlay_embed,
            build_single_pick_embed,
        )

        sport = self.sport

        try:
            with get_sync_session() as session:
                parlay = build_parlay(session, sport, "draftkings", n_legs)
        except Exception as exc:
            log.error("discord.build_parlay_failed", error=str(exc))
            await interaction.followup.send(
                "⚠️ Error generating picks. Try again in a moment.", ephemeral=True
            )
            return

        if parlay is None:
            embed = build_no_picks_embed(sport, n_legs)
            await interaction.followup.send(embed=discord.Embed.from_dict(embed), ephemeral=True)
            return

        requested_by = interaction.user.display_name

        if n_legs == 1:
            embed_dict = build_single_pick_embed(parlay.legs[0], sport, requested_by)
        else:
            embed_dict = build_parlay_embed(parlay, sport, requested_by)

        confirm_view = ConfirmParlayView(parlay=parlay, sport=sport)
        await interaction.followup.send(
            embed=discord.Embed.from_dict(embed_dict),
            view=confirm_view,
        )

    @discord.ui.button(label="1-Leg Straight", style=discord.ButtonStyle.primary, emoji="1️⃣")
    async def one_leg(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._generate(interaction, 1)

    @discord.ui.button(label="3-Leg Parlay", style=discord.ButtonStyle.primary, emoji="3️⃣")
    async def three_leg(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._generate(interaction, 3)

    @discord.ui.button(label="5-Leg Parlay", style=discord.ButtonStyle.primary, emoji="5️⃣")
    async def five_leg(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._generate(interaction, 5)


class ConfirmParlayView(discord.ui.View):
    """Attached to the posted parlay embed. 'Lock It In' saves it to DB for tracking."""

    def __init__(self, parlay: object, sport: str) -> None:
        super().__init__(timeout=300)
        self.parlay = parlay
        self.sport = sport

    @discord.ui.button(
        label="✅ Lock It In",
        style=discord.ButtonStyle.success,
        custom_id="confirm_parlay",
    )
    async def lock_in(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)

        from src.db.models.discord_parlay import DiscordParlay
        from src.db.session import get_sync_session

        parlay = self.parlay
        try:
            with get_sync_session() as session:
                record = DiscordParlay(
                    discord_user_id=str(interaction.user.id),
                    discord_username=interaction.user.display_name,
                    discord_message_id=str(interaction.message.id) if interaction.message else None,
                    discord_channel_id=str(interaction.channel_id),
                    sport_code=self.sport,
                    bookmaker=parlay.legs[0].bookmaker if parlay.legs else "unknown",
                    n_legs=parlay.n_legs,
                    legs=[leg.to_dict() for leg in parlay.legs],
                    parlay_odds_american=parlay.parlay_odds_american,
                    parlay_ev=round(parlay.ev_per_100, 2),
                    status="pending",
                )
                session.add(record)
                session.commit()

            await interaction.followup.send(
                "🔒 Parlay locked! I'll post here when results come in.", ephemeral=True
            )
            button.disabled = True
            button.label = "✅ Locked"
            await interaction.message.edit(view=self)

        except Exception as exc:
            log.error("discord.lock_in_failed", error=str(exc))
            await interaction.followup.send("⚠️ Couldn't save parlay. Try again.", ephemeral=True)

    @discord.ui.button(
        label="🔄 New Picks",
        style=discord.ButtonStyle.secondary,
        custom_id="regen_parlay",
    )
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = SportSelectView()
        await interaction.response.edit_message(
            content="🎯 **Pick a sport to get new predictions:**",
            embed=None,
            view=view,
        )


# ── /kirkova flow ─────────────────────────────────────────────────────────────


class KirkovaView(discord.ui.View):
    """Shows all today's picks as a multi-select. User locks in the ones they want."""

    def __init__(self, legs: list[ParlayLeg]) -> None:
        super().__init__(timeout=300)
        self.legs = legs
        self.selected_indices: list[int] = []

        options: list[discord.SelectOption] = []
        for i, leg in enumerate(legs[:25]):
            pick_team = leg.home_team if leg.pick == "home" else leg.away_team
            opp_team = leg.away_team if leg.pick == "home" else leg.home_team
            odds_str = f"+{leg.odds_american}" if leg.odds_american > 0 else str(leg.odds_american)
            sport_icon = "🏀" if leg.sport_code == "nba" else "⚾"
            options.append(
                discord.SelectOption(
                    label=f"✅ {sport_icon} {pick_team} ({odds_str})",
                    description=f"vs {opp_team}  ·  {leg.model_prob:.0%} model  ·  edge {leg.edge:+.1%}",
                    value=str(i),
                )
            )

        self.pick_select = discord.ui.Select(
            placeholder="Choose picks to lock in…",
            options=options,
            min_values=1,
            max_values=min(len(options), 25),
            custom_id="kirkova_select",
        )
        self.pick_select.callback = self._on_select
        self.add_item(self.pick_select)

    async def _on_select(self, interaction: discord.Interaction) -> None:
        self.selected_indices = [int(v) for v in self.pick_select.values]
        selected = [self.legs[i] for i in self.selected_indices]
        lines = []
        for leg in selected:
            pick_team = leg.home_team if leg.pick == "home" else leg.away_team
            opp_team = leg.away_team if leg.pick == "home" else leg.home_team
            odds_str = f"+{leg.odds_american}" if leg.odds_american > 0 else str(leg.odds_american)
            lines.append(f"• **{pick_team}** ({odds_str}) vs {opp_team}")
        preview = "\n".join(lines)
        await interaction.response.edit_message(
            content=f"**{len(selected)} pick(s) selected:**\n{preview}\n\nClick 🔒 to lock these in.",
            view=self,
        )

    @discord.ui.button(label="🔒 Lock Selected", style=discord.ButtonStyle.success)
    async def lock_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        if not self.selected_indices:
            await interaction.response.send_message(
                "Select at least one pick first!", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        from src.db.models.discord_parlay import DiscordParlay
        from src.db.session import get_sync_session
        from src.models.parlay import parlay_ev
        from src.notify.discord_embeds import build_kirkova_locked_embed

        selected_legs = [self.legs[i] for i in self.selected_indices]

        try:
            with get_sync_session() as session:
                for leg in selected_legs:
                    record = DiscordParlay(
                        discord_user_id=str(interaction.user.id),
                        discord_username=interaction.user.display_name,
                        discord_message_id=str(interaction.message.id)
                        if interaction.message
                        else None,
                        discord_channel_id=str(interaction.channel_id),
                        sport_code=leg.sport_code,
                        bookmaker=leg.bookmaker,
                        n_legs=1,
                        legs=[leg.to_dict()],
                        parlay_odds_american=leg.odds_american,
                        parlay_ev=round(parlay_ev([leg]), 2),
                        status="pending",
                    )
                    session.add(record)
                session.commit()

            embed_dict = build_kirkova_locked_embed(selected_legs, interaction.user.display_name)
            await interaction.followup.send(embed=discord.Embed.from_dict(embed_dict))

            button.disabled = True
            button.label = f"✅ {len(selected_legs)} Pick(s) Locked"
            await interaction.message.edit(view=self)

        except Exception as exc:
            log.error("discord.kirkova_lock_failed", error=str(exc))
            await interaction.followup.send("⚠️ Couldn't save picks. Try again.", ephemeral=True)
