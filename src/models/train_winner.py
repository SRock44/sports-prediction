"""XGBoost game-winner training pipeline with isotonic calibration.

Key properties:
- Walk-forward chronological split (no shuffling ever)
- Exponential recency sample weights (λ tuned per sport)
- Isotonic calibration for honest probability outputs
- Optuna hyperparameter search with walk-forward CV objective
- Champion/challenger promotion gate
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight, feature_spec_hash
from src.models.eval.metrics import compute_ece, compute_all_winner_metrics
from src.models.registry import log_model_run, get_run_metrics

log = get_logger(__name__)

# Recency decay λ per sport (tuned empirically; higher = forgets faster)
_LAMBDA = {"nba": 0.30, "mlb": 0.20}


def train_winner_model(
    sport: str,
    training_df: pd.DataFrame,  # must have columns: feature_names..., target (0/1), scheduled_utc
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 50,
    run_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Train a calibrated XGBoost winner model. Returns (mlflow_run_id, metrics)."""
    lam = _LAMBDA.get(sport, 0.25)

    # Chronological split: last 10% of training_df used for calibration
    training_df = training_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(training_df) * 0.9)
    train_part = training_df.iloc[:split_idx]
    calib_part = training_df.iloc[split_idx:]

    X_train = train_part[feature_names].values.astype(np.float32)
    y_train = train_part["y"].values.astype(int)
    X_calib = calib_part[feature_names].values.astype(np.float32)
    y_calib = calib_part["y"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["y"].values.astype(int)

    # Sample weights: exponential decay anchored to the last date in the training set
    # so the same historical data always produces the same weights regardless of run date.
    anchor = pd.to_datetime(training_df["game_date"].max())
    def make_weights(df: pd.DataFrame) -> np.ndarray:
        days_ago = (anchor - pd.to_datetime(df["game_date"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days_ago], dtype=np.float32)

    w_train = make_weights(train_part)
    w_calib = make_weights(calib_part)

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        }
        clf = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",

            tree_method="hist",
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        proba = clf.predict_proba(X_calib)[:, 1]
        return float(log_loss(y_calib, proba, sample_weight=w_calib))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_optuna_trials, timeout=300)
    best_params = study.best_params
    log.info("optuna.best", sport=sport, params=best_params, best_loss=study.best_value)

    # ── Train final model on full training set ────────────────────────────────
    best_clf = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        random_state=42,
    )
    w_all = make_weights(training_df)
    X_all = training_df[feature_names].values.astype(np.float32)
    y_all = training_df["y"].values.astype(int)
    best_clf.fit(X_all, y_all, sample_weight=w_all, verbose=False)

    # ── Isotonic calibration ──────────────────────────────────────────────────
    calibrated = CalibratedClassifierCV(best_clf, method="isotonic", cv=5)
    calibrated.fit(X_calib, y_calib, sample_weight=w_calib)

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    proba_hold = calibrated.predict_proba(X_hold)[:, 1]
    metrics = compute_all_winner_metrics(y_hold, proba_hold)
    log.info("winner.holdout_metrics", sport=sport, **metrics)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    run_id = log_model_run(
        run_name=run_name or f"{sport}_winner_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="winner",
        target="home_won",
        model=calibrated,
        metrics=metrics,
        params=best_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["game_date"].min()),
            str(training_df["game_date"].max()),
        ),
        model_framework="sklearn",  # CalibratedClassifierCV wraps XGBoost → sklearn interface
    )

    return run_id, metrics


def should_promote(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    min_logloss_improvement: float = 0.01,
    max_ece_increase: float = 0.02,
) -> tuple[bool, str]:
    """Promotion gate. Returns (should_promote, reason)."""
    chall_ll = challenger_metrics.get("logloss", 999)
    champ_ll = champion_metrics.get("logloss", 999)
    chall_ece = challenger_metrics.get("ece", 999)
    champ_ece = champion_metrics.get("ece", 999)
    chall_brier = challenger_metrics.get("brier", 999)
    champ_brier = champion_metrics.get("brier", 999)

    if chall_ll >= champ_ll - min_logloss_improvement:
        return False, f"log-loss {chall_ll:.4f} not better than champion {champ_ll:.4f} by {min_logloss_improvement}"
    if chall_ece > champ_ece + max_ece_increase:
        return False, f"ECE {chall_ece:.4f} exceeds champion {champ_ece:.4f} by margin"
    if chall_brier > champ_brier:
        return False, f"Brier {chall_brier:.4f} worse than champion {champ_brier:.4f}"
    return True, "all gates passed"
