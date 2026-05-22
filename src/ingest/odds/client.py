"""Odds aggregator — polls DraftKings, FanDuel, Fanatics, and Kalshi directly.

No third-party aggregator (e.g. The Odds API). We hit the books directly:
  - DraftKings  — moneyline + spread (unofficial web API, no key needed)
  - FanDuel     — moneyline + spread (unofficial web API, no key needed)
  - Fanatics    — moneyline + spread (unofficial web API, no key needed)
  - Kalshi      — implied probability from YES/NO prediction market (documented public API)

Consensus logic: average available moneyline implied probs across all books.
Kalshi probability is stored separately — it's already a calibrated probability.
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
    """Fetch and merge lines from DraftKings, FanDuel, and Fanatics.

    Each returned game dict has:
      home_team, away_team, commence_time,
      dk_home_ml, dk_away_ml, dk_home_spread,
      fd_home_ml, fd_away_ml, fd_home_spread,
      fan_home_ml, fan_away_ml, fan_home_spread,
      consensus_home_implied_prob, consensus_away_implied_prob,
      consensus_home_spread
    """
    book_lines: dict[str, list[dict[str, Any]]] = {}

    books = {
        "dk": ("src.ingest.odds.draftkings", "get_game_lines"),
        "fd": ("src.ingest.odds.fanduel", "get_game_lines"),
        "fan": ("src.ingest.odds.fanatics", "get_game_lines"),
    }

    for key, (module_path, fn_name) in books.items():
        try:
            import importlib

            mod = importlib.import_module(module_path)
            lines = getattr(mod, fn_name)(sport_code)
            book_lines[key] = lines
            log.info(f"odds.{key}_fetched", sport=sport_code, n=len(lines))
        except Exception as exc:
            log.warning(f"odds.{key}_failed", sport=sport_code, error=str(exc))
            book_lines[key] = []

    merged = _merge_all_books(book_lines)
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


_BOOK_KEYS = ["dk", "fd", "fan"]


def _merge_all_books(
    book_lines: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge lines from all books into one game list, indexed by home team.

    For each game:
      - Stores per-book fields: {key}_home_ml, {key}_away_ml, {key}_home_spread
      - Computes consensus implied prob as average across all books with data
    """
    # Build a master index: normalized_home_team → merged game dict
    games: dict[str, dict[str, Any]] = {}

    # Use DK as the primary source to establish the game list, fall back to FD, then Fanatics
    primary_order = [k for k in _BOOK_KEYS if book_lines.get(k)]

    for key in primary_order:
        for game_data in book_lines[key]:
            home_key = _norm(game_data.get("home_team", ""))
            if not home_key:
                continue
            if home_key not in games:
                games[home_key] = {
                    "home_team": game_data.get("home_team", ""),
                    "away_team": game_data.get("away_team", ""),
                    "commence_time": game_data.get("commence_time", ""),
                    **{f"{k}_home_ml": None for k in _BOOK_KEYS},
                    **{f"{k}_away_ml": None for k in _BOOK_KEYS},
                    **{f"{k}_home_spread": None for k in _BOOK_KEYS},
                }
            # Populate this book's fields
            games[home_key][f"{key}_home_ml"] = game_data.get("home_ml")
            games[home_key][f"{key}_away_ml"] = game_data.get("away_ml")
            games[home_key][f"{key}_home_spread"] = game_data.get("home_spread")

    # Compute consensus stats for each game
    out = []
    for game in games.values():
        ml_probs = [
            american_to_implied_prob(game[f"{k}_home_ml"])
            for k in _BOOK_KEYS
            if game.get(f"{k}_home_ml") is not None
        ]
        spreads = [
            game[f"{k}_home_spread"] for k in _BOOK_KEYS if game.get(f"{k}_home_spread") is not None
        ]

        game["consensus_home_implied_prob"] = sum(ml_probs) / len(ml_probs) if ml_probs else 0.5
        game["consensus_away_implied_prob"] = 1.0 - game["consensus_home_implied_prob"]
        game["consensus_home_spread"] = sum(spreads) / len(spreads) if spreads else 0.0
        out.append(game)

    return out


def _norm(name: str) -> str:
    """Normalize team name for fuzzy matching: lowercase last word."""
    parts = name.strip().lower().split()
    return parts[-1] if parts else name.lower()
