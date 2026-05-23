"""Print a status report of what NBA data is currently in the database.

Shows per-season counts for games, box scores, and features so you can see
exactly how far along the gather process is at any time.

Usage:
  docker compose exec api python scripts/nba_data_status.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.time import nba_season_for_date
from src.db.models import Game, MatchupFeature, Player, PlayerGameStats, Sport, Team
from src.db.session import get_sync_session


def main() -> None:
    today = date.today()
    current_season = nba_season_for_date(today)
    season_years = list(range(current_season - 4, current_season + 1))

    with get_sync_session() as session:
        sport = session.query(Sport).filter_by(code="nba").first()
        if sport is None:
            print("No NBA data found. Run gather_nba_training_data.py first.")
            sys.exit(0)

        n_teams = session.query(Team).filter_by(sport_id=sport.id).count()
        n_players = session.query(Player).filter_by(sport_id=sport.id).count()

        print("\nNBA Database Status")
        print(f"{'─'*65}")
        print(f"  Teams:   {n_teams}")
        print(f"  Players: {n_players}")
        print()
        print(f"  {'Season':<10} {'Games':>8} {'Final':>8} {'Box Scores':>12} {'Features':>10} {'Coverage':>10}")
        print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*12} {'─'*10} {'─'*10}")

        total_games = total_final = total_bs = total_feat = 0

        for season_year in season_years:
            label = f"{season_year}-{str(season_year+1)[-2:]}"

            n_games = session.query(Game).filter(
                Game.sport_id == sport.id,
                Game.season == season_year,
            ).count()

            n_final = session.query(Game).filter(
                Game.sport_id == sport.id,
                Game.season == season_year,
                Game.status == "final",
            ).count()

            games_with_bs = (
                session.query(Game.id)
                .filter(
                    Game.sport_id == sport.id,
                    Game.season == season_year,
                    Game.status == "final",
                    Game.id.in_(
                        session.query(PlayerGameStats.game_id).distinct()
                    ),
                )
                .count()
            )

            games_with_feat = (
                session.query(MatchupFeature.game_id)
                .join(Game, MatchupFeature.game_id == Game.id)
                .filter(
                    Game.sport_id == sport.id,
                    Game.season == season_year,
                )
                .count()
            )

            coverage = f"{games_with_bs / n_final * 100:.0f}%" if n_final > 0 else "—"
            marker = " ← missing" if n_final > 0 and games_with_bs < n_final * 0.95 else ""

            print(
                f"  {label:<10} {n_games:>8} {n_final:>8} {games_with_bs:>12} "
                f"{games_with_feat:>10} {coverage:>10}{marker}"
            )

            total_games += n_games
            total_final += n_final
            total_bs += games_with_bs
            total_feat += games_with_feat

        print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*12} {'─'*10} {'─'*10}")
        total_cov = f"{total_bs / total_final * 100:.0f}%" if total_final > 0 else "—"
        print(
            f"  {'TOTAL':<10} {total_games:>8} {total_final:>8} {total_bs:>12} "
            f"{total_feat:>10} {total_cov:>10}"
        )

        print()

        # Readiness assessment
        ready_for_training = total_feat >= total_final * 0.90
        if ready_for_training:
            print("  ✓  Sufficient features to train. Run:")
            print("     python -m src.cli train --sport nba --kind winner --trials 50 --promote")
        else:
            missing_bs = total_final - total_bs
            missing_feat = total_final - total_feat
            if missing_bs > 0:
                print(f"  ✗  Missing box scores for {missing_bs} games. Run:")
                print("     python scripts/gather_nba_training_data.py --box-scores-only")
            if missing_feat > 0:
                print(f"  ✗  Missing features for {missing_feat} games. Run:")
                print("     python scripts/build_nba_features.py")

        print()


if __name__ == "__main__":
    main()
