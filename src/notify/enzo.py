"""Data layer for /enzo — objective winner predictions with mathematical explanations.

No odds language. Just: who wins, why, with numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

_EDT = timezone(timedelta(hours=-4))

_FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "Elo gap",
    "elo_diff_with_hca": "Elo + home-court adj",
    "elo_home_win_prob": "Elo baseline",
    "net_rtg_diff_last5": "Net RTG diff (L5)",
    "net_rtg_diff_last10": "Net RTG diff (L10)",
    "net_rtg_diff_last20": "Net RTG diff (L20)",
    "win_pct_diff_last5": "Win% diff (L5)",
    "win_pct_diff_last10": "Win% diff (L10)",
    "win_pct_diff_last20": "Win% diff (L20)",
    "b2b_diff": "Back-to-back",
    "rest_diff": "Rest days",
    "starter_avail_diff": "Starter availability",
    "margin_diff_last10": "Scoring margin (L10)",
    "margin_diff_last5": "Scoring margin (L5)",
    "streak_diff": "Streak diff",
    "h2h_home_win_pct": "H2H win%",
    "away_travel_km": "Away travel (km)",
    "schedule_load_diff": "Schedule load",
    "roster_star_pts_diff": "Star player pts diff",
    "pitcher_era_diff": "Pitcher ERA diff",
    "ops_diff_last10": "OPS diff (L10)",
    "woba_diff_last10": "wOBA diff (L10)",
    "bullpen_era_diff": "Bullpen ERA diff",
    "home_ops_diff": "Home OPS diff",
    "away_ops_diff": "Away OPS diff",
}


@dataclass
class EnzoGame:
    game_id: int
    sport_code: str
    home_team: str
    away_team: str
    scheduled_utc: datetime
    external_id: str = ""
    probability: float | None = None
    raw_features: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    dk_home_price: float | None = None
    dk_away_price: float | None = None
    has_features: bool = False


def fetch_enzo_games(session: Session) -> list[EnzoGame]:
    """Return all upcoming games (next 48h) with predictions, features, and odds."""
    rows = session.execute(
        text("""
            SELECT
                g.id              AS game_id,
                sp.code           AS sport_code,
                ht.name           AS home_team,
                at_t.name         AS away_team,
                g.scheduled_utc,
                COALESCE(g.external_id, '') AS external_id,
                p.probability,
                pa.raw_features,
                pa.model_version,
                go.home_price     AS dk_home,
                go.away_price     AS dk_away,
                (mf.game_id IS NOT NULL) AS has_features
            FROM games g
            JOIN sports sp        ON sp.id = g.sport_id
            JOIN teams ht         ON ht.id = g.home_team_id
            JOIN teams at_t       ON at_t.id = g.away_team_id
            LEFT JOIN predictions p
                ON p.game_id = g.id AND p.target = 'home_won'
            LEFT JOIN predictions_audit pa
                ON pa.prediction_id = p.id
            LEFT JOIN game_odds go
                ON go.game_id = g.id
               AND go.market = 'h2h'
               AND go.bookmaker = 'draftkings'
            LEFT JOIN matchup_features mf
                ON mf.game_id = g.id
            WHERE g.scheduled_utc > NOW()
              AND g.scheduled_utc < NOW() + INTERVAL '48 hours'
              AND g.status = 'scheduled'
            ORDER BY sp.code, g.scheduled_utc
        """)
    ).fetchall()

    seen: set[int] = set()
    games: list[EnzoGame] = []
    for r in rows:
        if r.game_id in seen:
            continue
        seen.add(r.game_id)
        games.append(
            EnzoGame(
                game_id=r.game_id,
                sport_code=r.sport_code,
                home_team=r.home_team,
                away_team=r.away_team,
                scheduled_utc=r.scheduled_utc,
                external_id=r.external_id or "",
                probability=float(r.probability) if r.probability is not None else None,
                raw_features=dict(r.raw_features) if r.raw_features else {},
                model_version=r.model_version,
                dk_home_price=float(r.dk_home) if r.dk_home is not None else None,
                dk_away_price=float(r.dk_away) if r.dk_away is not None else None,
                has_features=bool(r.has_features),
            )
        )
    return games


def _label(feat: str) -> str:
    return _FEATURE_LABELS.get(feat, feat.replace("_", " "))


def _odds_pct(american: float) -> float:
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def _odds_fmt(american: float) -> str:
    return f"+{american:.0f}" if american > 0 else f"{american:.0f}"


def format_game_field(game: EnzoGame) -> tuple[str, str]:
    """Return (field_name, field_value) for a single game Discord embed field."""
    sport_icon = "🏀" if game.sport_code == "nba" else "⚾"
    local = game.scheduled_utc.astimezone(_EDT)
    time_str = local.strftime("%a %b %-d  %-I:%M %p ET")
    name = f"{sport_icon} {game.away_team} @ {game.home_team}  ·  {time_str}"

    # ── No prediction ─────────────────────────────────────────────────────────
    if game.probability is None:
        if not game.has_features:
            reason = "matchup features not yet computed — score task runs every 30 min"
        elif game.external_id.startswith("odds_"):
            reason = "game was just seeded from odds data — scoring task pending"
        else:
            reason = "prediction not generated yet — no active model or score task hasn't run"
        return name, f"❓ **Cannot predict**\n_Why: {reason}_"

    # ── Build prediction ──────────────────────────────────────────────────────
    prob = game.probability
    pick = "home" if prob >= 0.5 else "away"
    pick_team = game.home_team if pick == "home" else game.away_team
    pick_prob = prob if pick == "home" else 1.0 - prob
    conf = "🟢" if pick_prob >= 0.65 else "🟡" if pick_prob >= 0.58 else "⚪"

    raw = game.raw_features
    contribs: dict[str, float] = raw.get("_contribs", {})

    lines: list[str] = []

    # Verdict
    lines.append(f"{conf} **{pick_team}** wins  ·  `{pick_prob:.1%}` model confidence")

    # ── Pure model math ───────────────────────────────────────────────────────
    lines.append("**📐 Model math (no odds):**")

    # Elo baseline
    elo_prob = raw.get("elo_home_win_prob")
    elo_diff = raw.get("elo_diff")
    if elo_prob is not None and elo_diff is not None:
        elo_pick = game.home_team if elo_prob >= 0.5 else game.away_team
        elo_pct = elo_prob if elo_prob >= 0.5 else 1.0 - elo_prob
        lines.append(f"  Elo → **{elo_pick}** {elo_pct:.1%}  (gap: {elo_diff:+.0f} pts)")

    # Net rating
    net10 = raw.get("net_rtg_diff_last10")
    h_net10 = raw.get("home_net_rtg_last10")
    a_net10 = raw.get("away_net_rtg_last10")
    if h_net10 is not None and a_net10 is not None:
        lines.append(
            f"  Net RTG (L10) → {game.home_team} {h_net10:+.1f} / {game.away_team} {a_net10:+.1f}"
            + (f"  (diff: {net10:+.1f})" if net10 is not None else "")
        )

    # Win%
    h_wp = raw.get("home_win_pct_last10")
    a_wp = raw.get("away_win_pct_last10")
    if h_wp is not None and a_wp is not None:
        lines.append(f"  Win% (L10) → {game.home_team} {h_wp:.0%} vs {game.away_team} {a_wp:.0%}")

    # Rest / B2B
    rest = raw.get("rest_diff")
    b2b = raw.get("b2b_diff")
    if rest is not None and abs(rest) >= 0.5:
        rest_team = game.home_team if rest > 0 else game.away_team
        lines.append(f"  Rest → {rest_team} +{abs(rest):.1f} days advantage")
    if b2b is not None and b2b != 0:
        b2b_hurt = game.away_team if b2b < 0 else game.home_team
        lines.append(f"  ⚠️ {b2b_hurt} on back-to-back")

    # MLB-specific
    pitcher_era = raw.get("pitcher_era_diff")
    if pitcher_era is not None:
        era_team = game.home_team if pitcher_era < 0 else game.away_team
        lines.append(f"  Pitcher ERA → {era_team} has better starter ({pitcher_era:+.2f})")
    woba = raw.get("woba_diff_last10")
    if woba is not None and abs(woba) > 0.005:
        woba_team = game.home_team if woba > 0 else game.away_team
        lines.append(f"  wOBA (L10) → {woba_team} leads by {abs(woba):.3f}")

    # Top SHAP contributions
    if contribs:
        sign = 1.0 if pick == "home" else -1.0
        top3 = sorted(contribs.items(), key=lambda kv: kv[1] * sign, reverse=True)[:3]
        parts = []
        for feat, val in top3:
            arrow = "▲" if val * sign > 0 else "▼"
            parts.append(f"{arrow} {_label(feat)} ({val * sign:+.3f})")
        lines.append(f"  Top SHAP: {' · '.join(parts)}")

    # ── Odds context (separate, clearly labeled) ──────────────────────────────
    if game.dk_home_price is not None and game.dk_away_price is not None:
        book_home_pct = _odds_pct(game.dk_home_price)
        market_pick = game.home_team if book_home_pct >= 0.5 else game.away_team
        market_pct = book_home_pct if book_home_pct >= 0.5 else 1.0 - book_home_pct
        agree = "✅ agrees" if market_pick == pick_team else "⚡ disagrees with market"
        lines.append(
            f"**📊 Market context:** DK {_odds_fmt(game.dk_home_price)}/{_odds_fmt(game.dk_away_price)}"
            f" · Market favors {market_pick} ({market_pct:.0%}) · Model {agree}"
        )

    return name, "\n".join(lines)


def build_enzo_embeds(games: list[EnzoGame]) -> list[dict[str, Any]]:
    """Build a list of Discord embed dicts — one per sport, paginated if needed."""
    by_sport: dict[str, list[EnzoGame]] = {"nba": [], "mlb": []}
    for g in games:
        by_sport.setdefault(g.sport_code, []).append(g)

    embeds: list[dict[str, Any]] = []

    sport_meta = {
        "nba": ("🏀", "NBA", 0x1D82B6),
        "mlb": ("⚾", "MLB", 0xE8473F),
    }

    for sport_code, sport_games in by_sport.items():
        if not sport_games:
            continue
        icon, label, color = sport_meta.get(sport_code, ("🏅", sport_code.upper(), 0x95A5A6))
        n_pred = sum(1 for g in sport_games if g.probability is not None)
        n_total = len(sport_games)

        # Paginate: 8 games per embed (keeps fields readable)
        chunks = [sport_games[i : i + 8] for i in range(0, n_total, 8)]
        for page_idx, chunk in enumerate(chunks):
            page_str = f"  (page {page_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            fields = []
            for game in chunk:
                fname, fvalue = format_game_field(game)
                # Discord field value limit: 1024 chars
                if len(fvalue) > 1020:
                    fvalue = fvalue[:1020] + "…"
                fields.append({"name": fname, "value": fvalue, "inline": False})

            embeds.append(
                {
                    "title": f"{icon} {label} — Winner Predictions{page_str}",
                    "description": (
                        f"**{n_pred}/{n_total}** games predicted  ·  "
                        f"Pure model output — no betting context\n"
                        f"_🟢 ≥65% · 🟡 58-65% · ⚪ <58%_"
                    ),
                    "color": color,
                    "fields": fields,
                    "footer": {"text": "Model predicts outcomes only — not financial advice"},
                }
            )

    if not embeds:
        embeds.append(
            {
                "title": "🔍 No Upcoming Games",
                "description": "No scheduled games found in the next 48 hours.",
                "color": 0x95A5A6,
            }
        )

    return embeds
