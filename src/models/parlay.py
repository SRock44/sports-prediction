"""Parlay selection, odds math, and EV calculation.

Two entry points:
  select_top_picks()   — daily automated 10-pick list (1-leg each)
  build_parlay()       — user-requested N-leg parlay (1 / 3 / 5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import utc_now
from src.db.models import Game, GameOdds, ModelRecord, Prediction, Sport

log = get_logger(__name__)

# Minimum model edge over book implied probability to include a leg
_MIN_EDGE = 0.04
# Higher bar when picking the side the model de-favors (book mispricing, not model conviction).
_MIN_EDGE_CONTRA = 0.06
# Minimum absolute confidence (model prob distance from 50%) per sport.
# MLB probabilities cluster near 50% — 3% is still meaningful confidence.
_MIN_CONF: dict[str, float] = {"nba": 0.07, "mlb": 0.03}


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ParlayLeg:
    game_id: int
    sport_code: str
    home_team: str
    away_team: str
    scheduled_utc: Any  # datetime
    pick: str  # 'home' | 'away'
    model_prob: float  # model's P(pick wins)
    implied_prob: float  # book's implied P(pick wins)
    odds_american: int  # moneyline for the pick side
    bookmaker: str
    edge: float  # model_prob - implied_prob
    confidence: float  # abs(model_prob - 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "sport_code": self.sport_code,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "scheduled_utc": self.scheduled_utc.isoformat()
            if hasattr(self.scheduled_utc, "isoformat")
            else str(self.scheduled_utc),
            "pick": self.pick,
            "model_prob": round(self.model_prob, 4),
            "implied_prob": round(self.implied_prob, 4),
            "odds_american": self.odds_american,
            "bookmaker": self.bookmaker,
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Parlay:
    legs: list[ParlayLeg] = field(default_factory=list)
    parlay_odds_american: int = 0
    win_probability: float = 0.0  # model's combined P(all legs win)
    ev_per_100: float = 0.0  # expected value on $100 bet

    @property
    def n_legs(self) -> int:
        return len(self.legs)


# ── Odds math ─────────────────────────────────────────────────────────────────


def american_to_decimal(odds: int) -> float:
    if odds >= 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def decimal_to_american(decimal: float) -> int:
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100)
    return round(-100.0 / (decimal - 1.0))


def american_to_implied(odds: int) -> float:
    """Convert American moneyline to raw implied probability (no vig removal)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def combine_parlay_odds(legs: list[ParlayLeg]) -> int:
    combined = 1.0
    for leg in legs:
        combined *= american_to_decimal(leg.odds_american)
    return decimal_to_american(combined)


def parlay_ev(legs: list[ParlayLeg]) -> float:
    """Expected value per $100 wagered using model probabilities."""
    win_prob = math.prod(leg.model_prob for leg in legs)
    combined_decimal = math.prod(american_to_decimal(leg.odds_american) for leg in legs)
    payout = (combined_decimal - 1.0) * 100.0
    return win_prob * payout - (1.0 - win_prob) * 100.0


# ── DB helpers ────────────────────────────────────────────────────────────────


def _fetch_candidate_legs(
    session: Session,
    sport_code: str,
    bookmaker: str,
    hours_ahead: int = 36,
) -> list[ParlayLeg]:
    """Return all scoreable legs for upcoming games, ranked by edge."""
    now = utc_now()
    window_end = now + timedelta(hours=hours_ahead)

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        return []

    # Active winner model for this sport
    model_record = (
        session.query(ModelRecord)
        .filter_by(sport_id=sport.id, kind="winner", target="home_won", active=True)
        .first()
    )
    if model_record is None:
        log.warning("parlay.no_active_model", sport=sport_code)
        return []

    _BETTABLE_STATUSES = [
        "scheduled",
        "pre-game",
        "in progress",
        "in_progress_inning_1",
        "delayed",
        "delayed start",
    ]
    # Upcoming and live games: live games look back up to 4h to catch in-progress ones
    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= now - timedelta(hours=4),
            Game.scheduled_utc <= window_end,
            Game.status.in_(_BETTABLE_STATUSES),
        )
        .all()
    )

    legs: list[ParlayLeg] = []

    for game in games:
        pred = (
            session.query(Prediction)
            .filter_by(
                game_id=game.id,
                model_id=model_record.id,
                target="home_won",
                player_id=None,
            )
            .first()
        )
        if pred is None or pred.probability is None:
            continue

        model_home_prob = float(pred.probability)

        # Get odds for this game from the preferred bookmaker
        odds_row = (
            session.query(GameOdds)
            .filter_by(game_id=game.id, bookmaker=bookmaker, market="h2h")
            .order_by(GameOdds.fetched_at.desc())
            .first()
        )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            # Fall back to any available bookmaker if preferred not found
            odds_row = (
                session.query(GameOdds)
                .filter_by(game_id=game.id, market="h2h")
                .order_by(GameOdds.fetched_at.desc())
                .first()
            )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            continue

        home_odds = int(odds_row.home_price)
        away_odds = int(odds_row.away_price)
        home_implied = american_to_implied(home_odds)
        away_implied = american_to_implied(away_odds)

        home_team = game.home_team.name if game.home_team else f"Home {game.id}"
        away_team = game.away_team.name if game.away_team else f"Away {game.id}"

        # Decide pick: whichever side has the larger positive edge
        home_edge = model_home_prob - home_implied
        away_edge = (1.0 - model_home_prob) - away_implied

        min_conf = _MIN_CONF.get(sport_code.lower(), 0.05)
        model_favors_home = model_home_prob >= 0.5 + min_conf
        model_favors_away = (1.0 - model_home_prob) >= 0.5 + min_conf

        home_qualifies = home_edge >= _MIN_EDGE and (
            model_favors_home or home_edge >= _MIN_EDGE_CONTRA
        )
        away_qualifies = away_edge >= _MIN_EDGE and (
            model_favors_away or away_edge >= _MIN_EDGE_CONTRA
        )

        if home_edge >= away_edge and home_qualifies:
            legs.append(
                ParlayLeg(
                    game_id=game.id,
                    sport_code=sport_code,
                    home_team=home_team,
                    away_team=away_team,
                    scheduled_utc=game.scheduled_utc,
                    pick="home",
                    model_prob=model_home_prob,
                    implied_prob=home_implied,
                    odds_american=home_odds,
                    bookmaker=odds_row.bookmaker,
                    edge=home_edge,
                    confidence=abs(model_home_prob - 0.5),
                )
            )
        elif away_edge > home_edge and away_qualifies:
            legs.append(
                ParlayLeg(
                    game_id=game.id,
                    sport_code=sport_code,
                    home_team=home_team,
                    away_team=away_team,
                    scheduled_utc=game.scheduled_utc,
                    pick="away",
                    model_prob=1.0 - model_home_prob,
                    implied_prob=away_implied,
                    odds_american=away_odds,
                    bookmaker=odds_row.bookmaker,
                    edge=away_edge,
                    confidence=abs(model_home_prob - 0.5),
                )
            )

    # Sort by edge desc (best value first), then confidence as tiebreak
    legs.sort(key=lambda x: (x.edge, x.confidence), reverse=True)
    return legs


# ── Public API ────────────────────────────────────────────────────────────────

_BETTABLE_STATUSES = [
    "scheduled",
    "pre-game",
    "in progress",
    "in_progress_inning_1",
    "delayed",
    "delayed start",
]


def fetch_all_picks(
    session: Session,
    sport_code: str,
    bookmaker: str = "draftkings",
    hours_ahead: int = 36,
) -> list[ParlayLeg]:
    """Return a ParlayLeg for every upcoming/live game that has a prediction + odds.

    No edge or confidence filtering — always picks the model-favored side.
    Use this for display/selection UIs where the user chooses their own picks.
    """
    now = utc_now()
    window_end = now + timedelta(hours=hours_ahead)

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        return []

    model_record = (
        session.query(ModelRecord)
        .filter_by(sport_id=sport.id, kind="winner", target="home_won", active=True)
        .first()
    )
    if model_record is None:
        return []

    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= now - timedelta(hours=4),
            Game.scheduled_utc <= window_end,
            Game.status.in_(_BETTABLE_STATUSES),
        )
        .all()
    )

    legs: list[ParlayLeg] = []
    for game in games:
        pred = (
            session.query(Prediction)
            .filter_by(
                game_id=game.id,
                model_id=model_record.id,
                target="home_won",
                player_id=None,
            )
            .first()
        )
        if pred is None or pred.probability is None:
            continue

        model_home_prob = float(pred.probability)

        odds_row = (
            session.query(GameOdds)
            .filter_by(game_id=game.id, bookmaker=bookmaker, market="h2h")
            .order_by(GameOdds.fetched_at.desc())
            .first()
        )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            odds_row = (
                session.query(GameOdds)
                .filter_by(game_id=game.id, market="h2h")
                .order_by(GameOdds.fetched_at.desc())
                .first()
            )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            continue

        home_odds = int(odds_row.home_price)
        away_odds = int(odds_row.away_price)
        home_implied = american_to_implied(home_odds)
        away_implied = american_to_implied(away_odds)
        home_team = game.home_team.name if game.home_team else f"Home {game.id}"
        away_team = game.away_team.name if game.away_team else f"Away {game.id}"

        if model_home_prob >= 0.5:
            legs.append(
                ParlayLeg(
                    game_id=game.id,
                    sport_code=sport_code,
                    home_team=home_team,
                    away_team=away_team,
                    scheduled_utc=game.scheduled_utc,
                    pick="home",
                    model_prob=model_home_prob,
                    implied_prob=home_implied,
                    odds_american=home_odds,
                    bookmaker=odds_row.bookmaker,
                    edge=model_home_prob - home_implied,
                    confidence=abs(model_home_prob - 0.5),
                )
            )
        else:
            legs.append(
                ParlayLeg(
                    game_id=game.id,
                    sport_code=sport_code,
                    home_team=home_team,
                    away_team=away_team,
                    scheduled_utc=game.scheduled_utc,
                    pick="away",
                    model_prob=1.0 - model_home_prob,
                    implied_prob=away_implied,
                    odds_american=away_odds,
                    bookmaker=odds_row.bookmaker,
                    edge=(1.0 - model_home_prob) - away_implied,
                    confidence=abs(model_home_prob - 0.5),
                )
            )

    legs.sort(key=lambda x: x.model_prob, reverse=True)
    return legs


def select_top_picks(
    session: Session,
    sport_code: str,
    bookmaker: str = "draftkings",
    n: int = 10,
) -> list[ParlayLeg]:
    """Return up to n best single-game picks for today's proactive post."""
    legs = _fetch_candidate_legs(session, sport_code, bookmaker)
    return legs[:n]


def build_parlay(
    session: Session,
    sport_code: str,
    bookmaker: str,
    n_legs: int,
) -> Parlay | None:
    """Build an N-leg parlay from the top-confidence, best-edge games.

    Returns None if there aren't enough qualifying legs.
    """
    if n_legs not in (1, 3, 5):
        raise ValueError(f"n_legs must be 1, 3, or 5 — got {n_legs}")

    legs = _fetch_candidate_legs(session, sport_code, bookmaker)

    if len(legs) < n_legs:
        log.info(
            "parlay.insufficient_legs",
            sport=sport_code,
            wanted=n_legs,
            found=len(legs),
        )
        return None

    selected = legs[:n_legs]
    parlay = Parlay(
        legs=selected,
        parlay_odds_american=combine_parlay_odds(selected),
        win_probability=math.prod(leg.model_prob for leg in selected),
        ev_per_100=parlay_ev(selected),
    )
    return parlay
