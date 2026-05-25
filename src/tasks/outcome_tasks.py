"""Outcome tracking — checks completed games against predictions, fires Discord on wins.

Runs every 5 minutes. Uses Redis to avoid double-posting (TTL 48h per game).
"""

from __future__ import annotations

from typing import Any

import redis as redis_lib

from src.core.config import settings
from src.core.logging import get_logger
from src.db.session import sync_session_factory

log = get_logger(__name__)

_REDIS_PREFIX = "outcome:notified:"
_REDIS_TTL = 172_800  # 48 hours


# ── Human-readable feature labels ────────────────────────────────────────────

_FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "Elo rating gap",
    "elo_home_win_prob": "Elo implied win %",
    "elo_diff_with_hca": "Elo + home-court adj.",
    "net_rtg_diff_last5": "Net rating diff (L5)",
    "net_rtg_diff_last10": "Net rating diff (L10)",
    "net_rtg_diff_last20": "Net rating diff (L20)",
    "win_pct_diff_last10": "Win% diff (L10)",
    "win_pct_diff_last5": "Win% diff (L5)",
    "b2b_diff": "Back-to-back edge",
    "home_b2b": "Home on B2B",
    "away_b2b": "Away on B2B",
    "away_travel_km": "Away travel distance",
    "rest_diff": "Rest days edge",
    "starter_avail_diff": "Starter availability edge",
    "home_win_pct_season": "Home win% (season)",
    "away_win_pct_season": "Away win% (season)",
    "home_net_rtg_last10": "Home net rating (L10)",
    "away_net_rtg_last10": "Away net rating (L10)",
    "schedule_load_diff": "Schedule load diff",
    "streak_diff": "Win-streak diff",
    "h2h_home_win_pct": "H2H home win %",
    "odds_line_move": "Line move (sharp $)",
    "odds_sharp_move": "Sharp-money indicator",
    "sos_diff": "Strength of schedule diff",
    "home_sos_last10": "Home SOS (L10 Elo)",
    "away_sos_last10": "Away SOS (L10 Elo)",
    "margin_diff_last5": "Point diff margin (L5)",
    "margin_diff_last10": "Point diff margin (L10)",
}


def _label(feat: str) -> str:
    return _FEATURE_LABELS.get(feat, feat.replace("_", " ").title())


def _format_contrib(feat: str, value: float) -> str:
    direction = "▲" if value > 0 else "▼"
    return f"{direction} **{_label(feat)}**: {value:+.3f}"


# ── Core check logic ──────────────────────────────────────────────────────────


def check_outcomes(sport: str) -> dict[str, Any]:
    """Find newly completed games with pending predictions and post wins to Discord."""
    r = redis_lib.from_url(settings.redis_url, decode_responses=True)

    resolved = 0
    wins_posted = 0

    with sync_session_factory() as session:
        from sqlalchemy import text

        rows = session.execute(
            text("""
                SELECT
                    p.id          AS pred_id,
                    p.game_id,
                    p.probability,
                    p.model_id,
                    pa.raw_features,
                    pa.model_version,
                    g.home_score,
                    g.away_score,
                    g.scheduled_utc,
                    ht.name       AS home_team,
                    at_t.name     AS away_team,
                    sp.code       AS sport_code
                FROM predictions p
                JOIN games g       ON g.id = p.game_id
                JOIN sports sp     ON sp.id = g.sport_id
                JOIN teams ht      ON ht.id = g.home_team_id
                JOIN teams at_t    ON at_t.id = g.away_team_id
                LEFT JOIN predictions_audit pa ON pa.prediction_id = p.id
                WHERE sp.code = :sport
                  AND g.status = 'final'
                  AND p.target = 'home_won'
                  AND g.scheduled_utc > NOW() - INTERVAL '72 hours'
                ORDER BY g.scheduled_utc DESC
            """),
            {"sport": sport},
        )
        # SQLAlchemy alias trick — at_t alias maps to away_team column
        rows = list(rows)

    for row in rows:
        redis_key = f"{_REDIS_PREFIX}{row.game_id}"
        if r.exists(redis_key):
            continue  # already posted

        if row.home_score is None or row.away_score is None:
            continue

        actual_winner = "home" if row.home_score > row.away_score else "away"
        model_pick = "home" if float(row.probability) >= 0.5 else "away"
        model_prob = (
            float(row.probability) if model_pick == "home" else 1.0 - float(row.probability)
        )
        correct = actual_winner == model_pick

        # Mark as notified regardless of outcome (so we never double-post)
        r.setex(redis_key, _REDIS_TTL, "1")
        resolved += 1

        if correct:
            wins_posted += 1
            _post_win(row, model_prob, model_pick)

        # Also resolve any DiscordParlay records that include this game
        _resolve_parlay_legs(row.game_id, actual_winner, correct)

    log.info(
        "outcomes.check_done",
        sport=sport,
        resolved=resolved,
        wins_posted=wins_posted,
    )
    return {"resolved": resolved, "wins_posted": wins_posted}


def _post_win(row: Any, model_prob: float, model_pick: str) -> None:
    """Post a 'we called it' embed to the Discord results webhook."""
    webhook_url = (
        settings.discord_webhook_nba if row.sport_code == "nba" else settings.discord_webhook_mlb
    )
    if not webhook_url:
        return

    home_score = row.home_score
    away_score = row.away_score
    pick_team = row.home_team if model_pick == "home" else row.away_team
    score_str = f"{row.home_team} {home_score} - {away_score} {row.away_team}"

    # Pull feature contributions from audit
    raw = row.raw_features or {}
    contribs: dict[str, float] = raw.get("_contribs", {})
    # Show top 5 by absolute magnitude, filtered to same direction as the pick
    # (positive = home, negative = away; flip sign if we picked away)
    sign = 1.0 if model_pick == "home" else -1.0
    aligned = {k: v * sign for k, v in contribs.items()}
    top5 = sorted(aligned.items(), key=lambda kv: kv[1], reverse=True)[:5]

    contrib_lines = [_format_contrib(k, v) for k, v in top5] if top5 else ["_No feature data_"]

    # Season record from Redis (best-effort)
    record_str = _get_season_record(row.sport_code)

    embed = {
        "title": f"✅ CALLED IT — {row.away_team} @ {row.home_team}",
        "description": "\n".join(
            [
                f"**Result:** {score_str}",
                f"**Pick:** {pick_team}  ·  **Model confidence:** {model_prob:.1%}",
                "",
                "**🔬 Why the model knew:**",
                *contrib_lines,
                "",
                f"**📊 Season record:** {record_str}",
            ]
        ),
        "color": 0x2ECC71,  # green
        "footer": {"text": f"model v{row.model_version}  ·  {row.sport_code.upper()}"},
    }

    try:
        import httpx

        httpx.post(webhook_url, json={"embeds": [embed]}, timeout=10).raise_for_status()
        log.info("outcomes.win_posted", game_id=row.game_id, pick=pick_team)
    except Exception as exc:
        log.warning("outcomes.post_failed", game_id=row.game_id, error=str(exc))


def _resolve_parlay_legs(game_id: int, actual_winner: str, correct: bool) -> None:
    """Update DiscordParlay rows that contain this game as a leg."""
    from sqlalchemy.orm.attributes import flag_modified

    with sync_session_factory() as session:
        from src.db.models.discord_parlay import DiscordParlay

        parlays = (
            session.query(DiscordParlay)
            .filter(
                DiscordParlay.status == "pending",
                DiscordParlay.legs.contains([{"game_id": game_id}]),
            )
            .all()
        )

        for parlay in parlays:
            legs = parlay.legs or []
            for leg in legs:
                if leg.get("game_id") == game_id:
                    # Use the leg's own pick, not the model's raw probability direction.
                    # The bot may have picked the contra side (book mispricing), so
                    # the outer `correct` (model-prob vs actual) would be wrong here.
                    leg_correct = leg.get("pick") == actual_winner
                    leg["result"] = "won" if leg_correct else "lost"
                    leg["actual_winner"] = actual_winner

            # JSONB is not a MutableList — SQLAlchemy won't detect in-place dict
            # mutations. flag_modified forces the column into the UPDATE statement.
            parlay.legs = legs
            flag_modified(parlay, "legs")

            all_resolved = all(leg.get("result") for leg in legs)
            if all_resolved:
                n_correct = sum(1 for leg in legs if leg.get("result") == "won")
                parlay.n_correct = n_correct
                parlay.status = "won" if n_correct == len(legs) else "lost"
                from src.core.time import utc_now

                parlay.resolved_at = utc_now()

        session.commit()


def _get_season_record(sport: str) -> str:
    """Compute record across all resolved discord parlays, deduped by pick combination.

    Two parlays are the same pick if they cover the exact same games with the same
    pick direction — regardless of which user submitted them. Each unique pick
    combination counts once toward the record.
    """
    try:
        with sync_session_factory() as session:
            from sqlalchemy import text

            result = session.execute(
                text("""
                    WITH keyed AS (
                        SELECT
                            dp.status,
                            STRING_AGG(
                                (leg->>'game_id') || ':' || (leg->>'pick'),
                                ',' ORDER BY leg->>'game_id'
                            ) AS pick_key
                        FROM discord_parlays dp,
                        LATERAL jsonb_array_elements(dp.legs) AS leg
                        WHERE dp.sport_code = :sport
                          AND dp.status IN ('won', 'lost')
                          AND dp.created_at > NOW() - INTERVAL '180 days'
                        GROUP BY dp.id, dp.status
                    ),
                    unique_picks AS (
                        SELECT DISTINCT ON (pick_key) status
                        FROM keyed
                        ORDER BY pick_key
                    )
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'won') AS wins,
                        COUNT(*) AS total
                    FROM unique_picks
                """),
                {"sport": sport},
            ).first()
            if result and result.total > 0:
                pct = result.wins / result.total
                return f"{result.wins}-{result.total - result.wins} ({pct:.1%})"
    except Exception:
        pass
    return "N/A"


def post_daily_picks(sport: str) -> dict[str, Any]:
    """Post today's top-10 model picks to Discord. Runs at 14:00 UTC daily."""
    from src.db.models.discord_parlay import DiscordParlay
    from src.models.parlay import select_top_picks

    webhook_url = settings.discord_webhook_nba if sport == "nba" else settings.discord_webhook_mlb
    if not webhook_url:
        log.warning("daily_picks.no_webhook", sport=sport)
        return {"status": "skipped", "reason": "no_webhook"}

    with sync_session_factory() as session:
        # Try DraftKings first, fall back to FanDuel
        picks = select_top_picks(session, sport, bookmaker="draftkings", n=10)
        if len(picks) < 3:
            picks = select_top_picks(session, sport, bookmaker="fanduel", n=10)

        if not picks:
            log.info("daily_picks.no_qualifying", sport=sport)
            return {"status": "skipped", "reason": "no_qualifying_picks"}

        # Persist each pick as a 1-leg DiscordParlay so outcome task can track them
        for leg in picks:
            parlay = DiscordParlay(
                discord_user_id="bot",
                discord_username="PredictionBot",
                sport_code=sport,
                bookmaker=leg.bookmaker,
                n_legs=1,
                legs=[leg.to_dict()],
                parlay_odds_american=leg.odds_american,
                parlay_ev=round(leg.edge * 100, 2),
                status="pending",
            )
            session.add(parlay)
        session.commit()

    # Build the embed
    sport_emoji = "🏀" if sport == "nba" else "⚾"
    lines = [
        f"{sport_emoji} **TODAY'S TOP {len(picks)} MODEL PICKS** — powered by champion model",
        f"_Posted daily at 14:00 UTC  ·  {sport.upper()}_",
        "",
    ]

    for i, leg in enumerate(picks, 1):
        pick_team = leg.home_team if leg.pick == "home" else leg.away_team
        opp_team = leg.away_team if leg.pick == "home" else leg.home_team
        odds_str = f"+{leg.odds_american}" if leg.odds_american > 0 else str(leg.odds_american)
        game_time = (
            leg.scheduled_utc.strftime("%I:%M %p UTC")
            if hasattr(leg.scheduled_utc, "strftime")
            else ""
        )
        conf_bar = "🟢" if leg.model_prob >= 0.65 else "🟡" if leg.model_prob >= 0.58 else "⚪"
        lines.append(
            f"**{i}.** {conf_bar} **{pick_team}** to win vs {opp_team}  ·  _{game_time}_\n"
            f"   {odds_str}  ·  Model: **{leg.model_prob:.0%}**  ·  Edge: **{leg.edge:+.1%}**"
        )

    lines += [
        "",
        "_Wins posted immediately when results come in ✅_",
    ]

    embed = {
        "title": f"{sport_emoji} Daily Top Picks — {sport.upper()}",
        "description": "\n".join(lines),
        "color": 0x1D82B6 if sport == "nba" else 0xE8473F,
        "footer": {"text": "Edge = model prob - book implied prob  ·  Not financial advice"},
    }

    try:
        import httpx

        httpx.post(webhook_url, json={"embeds": [embed]}, timeout=10).raise_for_status()
        log.info("daily_picks.posted", sport=sport, n=len(picks))
    except Exception as exc:
        log.warning("daily_picks.post_failed", sport=sport, error=str(exc))
        return {"status": "error", "error": str(exc)}

    return {"status": "ok", "n_picks": len(picks)}


# ── Celery task registration ──────────────────────────────────────────────────

from src.tasks.celery_app import app  # noqa: E402

_post_daily_picks_fn = post_daily_picks


@app.task(name="src.tasks.outcome_tasks.check_outcomes_nba", bind=False)
def check_outcomes_nba() -> dict:
    return check_outcomes("nba")


@app.task(name="src.tasks.outcome_tasks.check_outcomes_mlb", bind=False)
def check_outcomes_mlb() -> dict:
    return check_outcomes("mlb")


@app.task(name="src.tasks.outcome_tasks.post_daily_picks", bind=False)
def post_daily_picks(sport: str) -> dict:  # type: ignore[misc]
    return _post_daily_picks_fn(sport)
