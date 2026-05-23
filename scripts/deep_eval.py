"""Deep accuracy evaluation of NBA and MLB champion models."""

from __future__ import annotations

import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")

from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402
from sqlalchemy import or_  # noqa: E402

from src.db.models import Game, MatchupFeature, Sport  # noqa: E402
from src.db.session import get_sync_session  # noqa: E402

NBA_PKL = (
    "/app/mlruns/707137705556388270/models/m-6dfea10f237b41e6ac3caa6bc69fda81/artifacts/model.pkl"
)
MLB_PKL = (
    "/app/mlruns/707137705556388270/models/m-392a5a3178484ef387a8d3443861014d/artifacts/model.pkl"
)
NBA_FEAT = (
    "/app/mlruns/707137705556388270/e23bd6156dfd4d96a5c94c3815f54e4f/artifacts/feature_names.json"
)
MLB_FEAT = (
    "/app/mlruns/707137705556388270/0dc9d02b36a744269f876df00bdeaab3/artifacts/feature_names.json"
)

with open(NBA_FEAT) as f:
    nba_features = json.load(f)["feature_names"]
with open(MLB_FEAT) as f:
    mlb_features = json.load(f)["feature_names"]
with open(NBA_PKL, "rb") as f:
    nba_model = pickle.load(f)
with open(MLB_PKL, "rb") as f:
    mlb_model = pickle.load(f)

# Move XGBoost predictor to CUDA for faster inference
for _label, _m in [("NBA", nba_model), ("MLB", mlb_model)]:
    try:
        _m.xgb_clf.get_booster().set_param({"device": "cuda"})
        print(f"{_label} XGB -> cuda OK")
    except Exception as e:
        print(f"{_label} XGB cuda failed: {e}")


def load_data(
    session: object,
    sport_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    sport_obj = session.query(Sport).filter_by(code=sport_code).first()
    rows = (
        session.query(Game, MatchupFeature)
        .join(MatchupFeature, MatchupFeature.game_id == Game.id)
        .filter(
            Game.sport_id == sport_obj.id,
            Game.status.in_(["final"]),
            or_(
                Game.meta["game_type"] == None,  # noqa: E711
                Game.meta["game_type"].astext == "R",
            ),
        )
        .order_by(Game.scheduled_utc)
        .all()
    )
    records = []
    for game, mf in rows:
        if not mf.features:
            continue
        rec = dict(mf.features)
        rec["y"] = 1 if (game.home_score or 0) > (game.away_score or 0) else 0
        rec["game_date"] = game.scheduled_utc.date() if game.scheduled_utc else None
        rec["season"] = game.season
        records.append(rec)
    df = pd.DataFrame(records).dropna(subset=["game_date"])
    mx = df["season"].max()
    return df[df["season"] < mx].copy(), df[df["season"] == mx].copy(), mx


def _ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y[mask].mean() - proba[mask].mean())
    return ece / len(y)


def deep_eval(
    model: object,
    holdout: pd.DataFrame,
    features: list[str],
    sport: str,
    season: object,
    train_n: int,
) -> None:
    for f in features:
        if f not in holdout.columns:
            holdout[f] = np.nan

    x = holdout[features].values.astype(np.float32)
    y = holdout["y"].values.astype(int)
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    conf = np.maximum(proba, 1 - proba)

    ll = log_loss(y, proba)
    brier = brier_score_loss(y, proba)
    acc = (pred == y).mean()
    coin_ll = log_loss(y, np.full(len(y), y.mean()))
    ece = _ece(y, proba)

    print(f"\n{'=' * 64}")
    print(
        f"  {sport.upper()} CHAMPION  |  Season {season} holdout  |  Trained on {train_n:,} games"
    )
    print(f"{'=' * 64}")
    print(f"  Accuracy        : {acc:.1%}  (home win rate in holdout: {y.mean():.1%})")
    print(f"  Log-loss        : {ll:.4f}  (coin-flip = {coin_ll:.4f}, edge = {coin_ll - ll:+.4f})")
    print(f"  Brier score     : {brier:.4f}  (lower=better, coin-flip ≈ 0.250)")
    print(f"  ECE             : {ece:.4f}  (calibration error, lower=better)")
    print(f"  N holdout games : {len(y):,}")

    if sport == "nba":
        print("\n  vs industry benchmarks (NBA):")
        print("    Random   : ~50%  log-loss ~0.693")
        print("    Elo only : ~60%  log-loss ~0.670")
        print("    Top sharp: ~65%  log-loss ~0.640")
        print(f"    Our model: {acc:.1%}  log-loss {ll:.4f}")
    else:
        print("\n  vs industry benchmarks (MLB):")
        print("    Random   : ~50%  log-loss ~0.693")
        print("    Elo only : ~54%  log-loss ~0.685")
        print("    Top sharp: ~57%  log-loss ~0.660")
        print(f"    Our model: {acc:.1%}  log-loss {ll:.4f}")

    print("\n  ACCURACY BY CONFIDENCE BUCKET")
    print(f"  {'Bucket':<16} {'N':>5} {'Acc':>7} {'AvgConf':>9} {'Edge vs 50%':>12}")
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]:
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 3:
            continue
        b_acc = (pred[mask] == y[mask]).mean()
        b_conf = conf[mask].mean()
        label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f">={lo:.0%}"
        print(
            f"  {label:<16} {mask.sum():>5}  {b_acc:>6.1%}  {b_conf:>8.1%}  {b_acc - 0.5:>+10.1%}"
        )

    print("\n  HIGH-CONFIDENCE PICKS (model certainty >= threshold)")
    print(f"  {'Min conf':<12} {'N':>5} {'Acc':>7} {'%AllGames':>10}")
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        mask = conf >= thresh
        if mask.sum() == 0:
            continue
        a = (pred[mask] == y[mask]).mean()
        print(f"  >={thresh:.0%}        {mask.sum():>5}  {a:>6.1%}  {mask.sum() / len(y):>9.1%}")

    print("\n  CALIBRATION  (are probabilities trustworthy?)")
    print(f"  {'Pred bin':<13} {'N':>5} {'Actual %':>9} {'Pred avg':>9} {'Gap':>7}")
    for lo in np.arange(0.25, 0.80, 0.05):
        hi = lo + 0.05
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() < 5:
            continue
        actual = y[mask].mean()
        pred_avg = proba[mask].mean()
        print(
            f"  {lo:.2f}-{hi:.2f}      {mask.sum():>5}  {actual:>8.1%}  {pred_avg:>8.1%}  {actual - pred_avg:>+6.1%}"
        )

    h = holdout.copy()
    h["_p"] = proba
    h["_ok"] = (pred == y).astype(int)
    h["_conf"] = conf
    h["_month"] = pd.to_datetime(h["game_date"]).dt.to_period("M")
    monthly = (
        h.groupby("_month")
        .agg(
            n=("_ok", "count"),
            acc=("_ok", "mean"),
            avg_conf=("_conf", "mean"),
        )
        .reset_index()
    )
    print(f"\n  MONTHLY BREAKDOWN (holdout season {season})")
    print(f"  {'Month':<10} {'N':>5} {'Acc':>7} {'AvgConf':>9}")
    for _, r in monthly.iterrows():
        print(f"  {r['_month']!s:<10} {int(r['n']):>5}  {r['acc']:>6.1%}  {r['avg_conf']:>8.1%}")

    xgb_trees = model.xgb_clf.get_booster().num_boosted_rounds()
    lgb_trees = model.lgb_clf.best_iteration_
    print(
        f"\n  Features: {len(features)}  |  XGB trees: {xgb_trees}  |  LGB trees: {lgb_trees}  |  LGB weight: {model.lgb_weight:.0%}"
    )


with get_sync_session() as session:
    nba_train, nba_hold, nba_season = load_data(session, "nba")
    mlb_train, mlb_hold, mlb_season = load_data(session, "mlb")

deep_eval(nba_model, nba_hold, nba_features, "nba", nba_season, len(nba_train))
deep_eval(mlb_model, mlb_hold, mlb_features, "mlb", mlb_season, len(mlb_train))
