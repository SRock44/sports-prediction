"""Quick health check: models, training data, predictions, upcoming games."""

from sqlalchemy import text

from src.db.models import ModelRecord, Sport
from src.db.session import sync_session_factory

with sync_session_factory() as s:
    for code in ("nba", "mlb"):
        sp = s.query(Sport).filter_by(code=code).first()
        if not sp:
            print(f"{code.upper()}: sport not found!")
            continue

        m = (
            s.query(ModelRecord)
            .filter_by(sport_id=sp.id, kind="winner", target="home_won", active=True)
            .first()
        )

        n_mf = s.execute(
            text(
                "SELECT COUNT(*) FROM matchup_features mf "
                "JOIN games g ON g.id=mf.game_id "
                "JOIN sports sp ON sp.id=g.sport_id WHERE sp.code=:c"
            ),
            {"c": code},
        ).scalar()

        n_pred = s.execute(
            text(
                "SELECT COUNT(*) FROM predictions p "
                "JOIN games g ON g.id=p.game_id "
                "JOIN sports sp ON sp.id=g.sport_id WHERE sp.code=:c"
            ),
            {"c": code},
        ).scalar()

        n_sched = s.execute(
            text(
                "SELECT COUNT(*) FROM games g "
                "JOIN sports sp ON sp.id=g.sport_id "
                "WHERE sp.code=:c AND g.status='scheduled' AND g.scheduled_utc > NOW()"
            ),
            {"c": code},
        ).scalar()

        n_today = s.execute(
            text(
                "SELECT COUNT(*) FROM predictions p "
                "JOIN games g ON g.id=p.game_id "
                "JOIN sports sp ON sp.id=g.sport_id "
                "WHERE sp.code=:c AND g.scheduled_utc > NOW() - INTERVAL '24 hours' "
                "AND g.scheduled_utc < NOW() + INTERVAL '24 hours'"
            ),
            {"c": code},
        ).scalar()

        model_str = f"{m.mlflow_run_id[:8]} v{m.version}" if m else "NO ACTIVE MODEL"
        print(f"\n{'=' * 40}")
        print(f"  {code.upper()}")
        print(f"  Champion model : {model_str}")
        print(f"  Matchup features: {n_mf:,} rows (training data)")
        print(f"  Total predictions: {n_pred:,}")
        print(f"  Today predictions: {n_today}")
        print(f"  Upcoming games in DB: {n_sched}")

    # Also check recent training logs
    import os

    log_path = "/app/reports/training_log.md"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print(f"\n{'=' * 40}")
        print("  Last 20 lines of training_log.md:")
        for line in lines[-20:]:
            print(" ", line.rstrip())
