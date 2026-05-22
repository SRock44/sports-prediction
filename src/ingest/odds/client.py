"""Odds aggregator — polls DraftKings, FanDuel, and Kalshi directly.

No third-party aggregator (e.g. The Odds API). We hit the books directly:
  - DraftKings  — moneyline + spread (unofficial web API, no key needed)
  - FanDuel     — moneyline + spread (unofficial web API, no key needed)
  - Kalshi      — implied probability from YES/NO prediction market (documented public API)

Consensus logic: average the two moneyline implied probs (DK + FD). Kalshi
probability is stored separately as it's already a probability, not odds.
"""
from __future__ import annotations

from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)


def american_to_implied_prob(price: float) -> float:
    """Convert American moneyline to implied probability (no vig removal)."""
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def get_consensus_lines(sport_code: str) -> list[dict[str, Any]]:
    """Fetch and merge lines from DraftKings + FanDuel. Returns list of game dicts.

    Each game dict has:
      home_team, away_team, commence_time,
      dk_home_ml, dk_away_ml, dk_home_spread,
      fd_home_ml, fd_away_ml, fd_home_spread,
      consensus_home_implied_prob, consensus_away_implied_prob,
      consensus_home_spread
    """
    dk_lines: list[dict[str, Any]] = []
    fd_lines: list[dict[str, Any]] = []

    try:
        from src.ingest.odds.draftkings import get_game_lines as dk_get
        dk_lines = dk_get(sport_code)
        log.info("odds.dk_fetched", sport=sport_code, n=len(dk_lines))
    except Exception as exc:
        log.warning("odds.dk_failed", sport=sport_code, error=str(exc))

    try:
        from src.ingest.odds.fanduel import get_game_lines as fd_get
        fd_lines = fd_get(sport_code)
        log.info("odds.fd_fetched", sport=sport_code, n=len(fd_lines))
    except Exception as exc:
        log.warning("odds.fd_failed", sport=sport_code, error=str(exc))

    # Merge by fuzzy team-name matching
    merged = _merge_lines(dk_lines, fd_lines)
    log.info("odds.merged", sport=sport_code, n=len(merged))
    return merged


def get_kalshi_probs(sport_code: str) -> list[dict[str, Any]]:
    """Fetch Kalshi prediction market probabilities separately."""
    try:
        from src.ingest.odds.kalshi import get_game_markets
        return get_game_markets(sport_code)
    except Exception as exc:
        log.warning("odds.kalshi_failed", sport=sport_code, error=str(exc))
        return []


def _merge_lines(
    dk: list[dict[str, Any]],
    fd: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair DK and FD lines for the same game, compute consensus implied prob."""
    out: list[dict[str, Any]] = []

    # Index FD by normalized home team name
    fd_index: dict[str, dict[str, Any]] = {
        _norm(g["home_team"]): g for g in fd if g.get("home_team")
    }

    for dk_game in dk:
        home = _norm(dk_game.get("home_team", ""))
        fd_game = fd_index.get(home)

        game: dict[str, Any] = {
            "home_team": dk_game.get("home_team", ""),
            "away_team": dk_game.get("away_team", ""),
            "commence_time": dk_game.get("commence_time", ""),
            "dk_home_ml": dk_game.get("home_ml"),
            "dk_away_ml": dk_game.get("away_ml"),
            "dk_home_spread": dk_game.get("home_spread"),
            "fd_home_ml": fd_game.get("home_ml") if fd_game else None,
            "fd_away_ml": fd_game.get("away_ml") if fd_game else None,
            "fd_home_spread": fd_game.get("home_spread") if fd_game else None,
        }

        # Consensus implied prob (average available books)
        probs = []
        for ml_key in ("dk_home_ml", "fd_home_ml"):
            ml = game.get(ml_key)
            if ml is not None:
                probs.append(american_to_implied_prob(ml))

        game["consensus_home_implied_prob"] = sum(probs) / len(probs) if probs else 0.5
        game["consensus_away_implied_prob"] = 1.0 - game["consensus_home_implied_prob"]

        spreads = [g for g in (game["dk_home_spread"], game["fd_home_spread"]) if g is not None]
        game["consensus_home_spread"] = sum(spreads) / len(spreads) if spreads else 0.0

        out.append(game)

    # Add any FD-only games not matched to DK
    matched_homes = {_norm(g["home_team"]) for g in out}
    for fd_game in fd:
        if _norm(fd_game.get("home_team", "")) not in matched_homes:
            game = {
                "home_team": fd_game.get("home_team", ""),
                "away_team": fd_game.get("away_team", ""),
                "commence_time": fd_game.get("commence_time", ""),
                "dk_home_ml": None, "dk_away_ml": None, "dk_home_spread": None,
                "fd_home_ml": fd_game.get("home_ml"),
                "fd_away_ml": fd_game.get("away_ml"),
                "fd_home_spread": fd_game.get("home_spread"),
            }
            ml = game["fd_home_ml"]
            game["consensus_home_implied_prob"] = american_to_implied_prob(ml) if ml else 0.5
            game["consensus_away_implied_prob"] = 1.0 - game["consensus_home_implied_prob"]
            game["consensus_home_spread"] = game["fd_home_spread"] or 0.0
            out.append(game)

    return out


def _norm(name: str) -> str:
    """Normalize team name for fuzzy matching: lowercase last word."""
    parts = name.strip().lower().split()
    return parts[-1] if parts else name.lower()
