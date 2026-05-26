"""Discord embed builders for predictions and parlays."""

from __future__ import annotations

from typing import Any

from src.models.parlay import Parlay, ParlayLeg, american_to_implied


def _odds_str(american: int) -> str:
    return f"+{american}" if american > 0 else str(american)


def _prob_bar(p: float, width: int = 12) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def build_parlay_embed(
    parlay: Parlay, sport: str, requested_by: str, risk_level: Any = None
) -> dict[str, Any]:
    """Rich embed for an N-leg parlay returned by /predict."""
    from src.models.parlay import DEFAULT_RISK, RISK_LEVELS

    sport_emoji = "🏀" if sport == "nba" else "⚾"
    color = 0x1D82B6 if sport == "nba" else 0xE8473F

    rl = risk_level or RISK_LEVELS[DEFAULT_RISK]
    risk_badge = f"{rl.emoji} {rl.name}"

    win_pct = parlay.win_probability * 100
    ev = parlay.ev_per_100
    ev_str = f"+${ev:.0f}" if ev >= 0 else f"-${abs(ev):.0f}"
    ev_color = "🟢" if ev >= 0 else "🔴"

    combined_odds = _odds_str(parlay.parlay_odds_american)
    implied = american_to_implied(parlay.parlay_odds_american)

    lines: list[str] = [
        f"**{parlay.n_legs}-Leg Parlay**  ·  {sport.upper()}  ·  {combined_odds}  ·  {risk_badge}",
        f"Book implied: {implied:.1%}  ·  Model win prob: **{win_pct:.1%}**",
        f"{ev_color} EV per $100: **{ev_str}**",
        "",
        "─────────────────────────────",
    ]

    for i, leg in enumerate(parlay.legs, 1):
        pick_team = leg.home_team if leg.pick == "home" else leg.away_team
        opp_team = leg.away_team if leg.pick == "home" else leg.home_team
        bar = _prob_bar(leg.model_prob)
        sched = (
            leg.scheduled_utc.strftime("%a %b %-d  %I:%M %p UTC")
            if hasattr(leg.scheduled_utc, "strftime")
            else ""
        )
        conf_icon = "🟢" if leg.model_prob >= 0.68 else "🟡" if leg.model_prob >= 0.60 else "⚪"
        lines += [
            f"**Leg {i}:** {conf_icon} {pick_team} to beat {opp_team}",
            f"`{bar}` **{leg.model_prob:.0%}**  ·  {_odds_str(leg.odds_american)}  ({leg.bookmaker})",
            f"Edge: **{leg.edge:+.1%}**  ·  _{sched}_",
            "",
        ]

    lines += [
        "─────────────────────────────",
        f"_Requested by {requested_by}_",
    ]

    return {
        "title": f"{sport_emoji} Parlay Pick",
        "description": "\n".join(lines),
        "color": color,
        "footer": {"text": "Press ✅ to track this parlay  ·  Not financial advice"},
    }


def build_single_pick_embed(
    leg: ParlayLeg, sport: str, requested_by: str, risk_level: Any = None
) -> dict[str, Any]:
    """Embed for a 1-leg straight bet from /predict."""
    from src.models.parlay import DEFAULT_RISK, RISK_LEVELS

    sport_emoji = "🏀" if sport == "nba" else "⚾"
    color = 0x1D82B6 if sport == "nba" else 0xE8473F
    pick_team = leg.home_team if leg.pick == "home" else leg.away_team
    opp_team = leg.away_team if leg.pick == "home" else leg.home_team
    bar = _prob_bar(leg.model_prob)
    sched = (
        leg.scheduled_utc.strftime("%a %b %-d  %I:%M %p UTC")
        if hasattr(leg.scheduled_utc, "strftime")
        else ""
    )
    ev = (
        leg.model_prob
        * (leg.odds_american / 100 if leg.odds_american > 0 else 100 / abs(leg.odds_american))
        - (1 - leg.model_prob)
    ) * 100

    rl = risk_level or RISK_LEVELS[DEFAULT_RISK]
    risk_badge = f"{rl.emoji} {rl.name}"
    conf_icon = "🟢" if leg.model_prob >= 0.68 else "🟡" if leg.model_prob >= 0.60 else "⚪"

    return {
        "title": f"{sport_emoji} Straight Pick  ·  {risk_badge}",
        "description": "\n".join(
            [
                f"{conf_icon} **{pick_team}** to beat **{opp_team}**",
                f"`{bar}` **{leg.model_prob:.0%}** model confidence",
                f"Odds: **{_odds_str(leg.odds_american)}** ({leg.bookmaker})",
                f"Book implied: {leg.implied_prob:.1%}  ·  Edge: **{leg.edge:+.1%}**",
                f"EV per $100: **{'+' if ev >= 0 else ''}{ev:.0f}**",
                f"_{sched}_",
                "",
                f"_Requested by {requested_by}_",
            ]
        ),
        "color": color,
        "footer": {"text": "Press ✅ to track  ·  Not financial advice"},
    }


def build_kirkova_embed(nba_legs: list[ParlayLeg], mlb_legs: list[ParlayLeg]) -> dict[str, Any]:
    """Full-day predictions embed for /kirkova."""
    lines: list[str] = []

    def _section(legs: list[ParlayLeg], sport: str) -> None:
        icon = "🏀" if sport == "nba" else "⚾"
        lines.append(
            f"{icon} **{sport.upper()}** — {len(legs)} value pick{'s' if len(legs) != 1 else ''}"
        )
        lines.append("──────────────────────")
        for leg in legs:
            pick_team = leg.home_team if leg.pick == "home" else leg.away_team
            opp_team = leg.away_team if leg.pick == "home" else leg.home_team
            odds_str = f"+{leg.odds_american}" if leg.odds_american > 0 else str(leg.odds_american)
            bar = _prob_bar(leg.model_prob, width=8)
            conf = "🟢" if leg.model_prob >= 0.65 else "🟡" if leg.model_prob >= 0.58 else "⚪"
            lines.append(f"{conf} **{pick_team}** to win vs {opp_team}")
            lines.append(
                f"`{bar}` {leg.model_prob:.0%}  ·  {odds_str}  ·  edge **{leg.edge:+.1%}**"
            )
        lines.append("")

    if nba_legs:
        _section(nba_legs, "nba")
    if mlb_legs:
        _section(mlb_legs, "mlb")

    total = len(nba_legs) + len(mlb_legs)
    return {
        "title": "✅ Today's Value Picks",
        "description": "\n".join(lines),
        "color": 0xF0B429,
        "footer": {
            "text": f"{total} pick{'s' if total != 1 else ''} with positive model edge  ·  Not financial advice"
        },
    }


def build_kirkova_locked_embed(legs: list[ParlayLeg], locked_by: str) -> dict[str, Any]:
    """Posted publicly when a user locks in picks via /kirkova."""
    lines: list[str] = [f"🔒 **{locked_by}** locked in {len(legs)} pick(s):", ""]
    for leg in legs:
        pick_team = leg.home_team if leg.pick == "home" else leg.away_team
        opp_team = leg.away_team if leg.pick == "home" else leg.home_team
        odds_str = f"+{leg.odds_american}" if leg.odds_american > 0 else str(leg.odds_american)
        sport_icon = "🏀" if leg.sport_code == "nba" else "⚾"
        lines.append(f"✅ {sport_icon} **{pick_team}** ({odds_str}) to beat {opp_team}")
        lines.append(f"  Model: **{leg.model_prob:.0%}**  ·  Edge: {leg.edge:+.1%}")

    return {
        "title": "📌 Picks Locked In",
        "description": "\n".join(lines),
        "color": 0x2ECC71,
        "footer": {"text": "Results posted when games finish  ·  Not financial advice"},
    }


def build_no_picks_embed(sport: str, n_legs: int) -> dict[str, Any]:
    return {
        "title": "😔 No Qualifying Picks",
        "description": "\n".join(
            [
                f"The model doesn't have **{n_legs}** games today with enough edge",
                f"over {sport.upper()} book lines to recommend a parlay.",
                "",
                "Try again later when more games are priced, or try a smaller parlay.",
            ]
        ),
        "color": 0x95A5A6,
    }


def _american_to_implied(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def build_props_embed(props: list[dict[str, Any]], sport: str) -> dict[str, Any]:
    """Embed showing top player prop edges for /props command.

    Each entry in props should have:
      player_name, target, dk_line, p_over, dk_over_implied, edge, model_median, game_label
    """
    sport_emoji = "🏀" if sport == "nba" else "⚾"
    color = 0x1D82B6 if sport == "nba" else 0xE8473F

    if not props:
        return {
            "title": f"{sport_emoji} No Props Available",
            "description": "No player prop predictions today, or models not yet trained.",
            "color": 0x95A5A6,
        }

    lines: list[str] = []
    for p in props:
        edge = p["edge"]
        direction = "OVER" if edge > 0 else "UNDER"
        abs_edge = abs(edge)
        edge_sign = "🟢" if abs_edge >= 0.05 else "🟡" if abs_edge >= 0.03 else "⚪"
        model_prob = p["p_over"] if edge > 0 else 1.0 - p["p_over"]
        odds_str = _odds_str(int(p.get("dk_odds", -110)))
        lines.append(
            f"{edge_sign} **{p['player_name']}** — {p['target']} {direction} {p['dk_line']:.1f}"
        )
        lines.append(
            f"  Model: **{model_prob:.0%}**  ·  Book: {p['dk_over_implied']:.0%}"
            f"  ·  Edge: **{abs_edge:+.1%}**  ·  {odds_str}"
        )
        lines.append(f"  Median projected: **{p['model_median']:.1f}**  ·  _{p['game_label']}_")
        lines.append("")

    return {
        "title": f"{sport_emoji} Today's Prop Edges",
        "description": "\n".join(lines).rstrip(),
        "color": color,
        "footer": {
            "text": f"{len(props)} prop{'s' if len(props) != 1 else ''} shown  ·  Not financial advice"
        },
    }
