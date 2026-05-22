"""XGBoost + LightGBM ensemble winner pipeline with manual isotonic calibration.

Key properties:
- Walk-forward 5-fold chronological CV in Optuna objective (no shuffling ever)
- 200 Optuna trials, MedianPruner, no timeout — full search
- XGBoost with n_estimators searched by Optuna; final fit uses early stopping to
  refine optimal tree count against calibration set
- Optional LightGBM soft ensemble (averaged 60/40 with XGBoost) if installed
- Manual isotonic regression calibration on prefit ensemble (no cv= leakage)
- Exponential recency sample weights anchored to training set end date
- Champion/challenger promotion gate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight, feature_spec_hash
from src.models.eval.metrics import compute_all_winner_metrics
from src.models.registry import log_model_run, get_run_metrics

log = get_logger(__name__)

_LAMBDA = {"nba": 0.30, "mlb": 0.20}

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


class IsotonicCalibratedEnsemble:
    """XGBoost (+ optional LightGBM) with manual isotonic probability calibration.

    Uses the prefit approach: the base models are trained on train_part, then raw
    ensemble probabilities on calib_part are isotonic-mapped to calibrated probs.
    Implements sklearn's predict_proba interface for MLflow pickle serialization.
    """

    def __init__(
        self,
        xgb_clf: xgb.XGBClassifier,
        iso: IsotonicRegression,
        lgb_clf=None,
        lgb_weight: float = 0.4,
    ) -> None:
        self.xgb_clf = xgb_clf
        self.lgb_clf = lgb_clf
        self.iso = iso
        self.lgb_weight = lgb_weight if lgb_clf is not None else 0.0
        self.classes_ = np.array([0, 1])

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_clf.predict_proba(X)[:, 1]
        if self.lgb_clf is not None:
            p_lgb = self.lgb_clf.predict_proba(X)[:, 1]
            return (1.0 - self.lgb_weight) * p_xgb + self.lgb_weight * p_lgb
        return p_xgb

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_proba(X)
        cal = np.clip(self.iso.predict(raw), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_cv_folds(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5
) -> list[tuple]:
    """Build expanding-window time-series CV folds (data must be sorted by date)."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        tr_end = i * fold_size
        val_end = (i + 1) * fold_size
        if val_end > n:
            break
        folds.append((
            X[:tr_end], y[:tr_end], w[:tr_end],
            X[tr_end:val_end], y[tr_end:val_end], w[tr_end:val_end],
        ))
    return folds


def train_winner_model(
    sport: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 200,
    run_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Train calibrated ensemble model. Returns (mlflow_run_id, metrics_dict)."""
    lam = _LAMBDA.get(sport, 0.25)

    training_df = training_df.sort_values("game_date").reset_index(drop=True)

    # Last 15% held for calibration; Optuna walks forward over the rest
    n = len(training_df)
    calib_start = int(n * 0.85)
    train_part = training_df.iloc[:calib_start]
    calib_part = training_df.iloc[calib_start:]

    X_train = train_part[feature_names].values.astype(np.float32)
    y_train = train_part["y"].values.astype(int)
    X_calib = calib_part[feature_names].values.astype(np.float32)
    y_calib = calib_part["y"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["y"].values.astype(int)

    anchor = pd.to_datetime(training_df["game_date"].max())

    def make_weights(df: pd.DataFrame) -> np.ndarray:
        days_ago = (anchor - pd.to_datetime(df["game_date"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days_ago], dtype=np.float32)

    w_train = make_weights(train_part)
    w_calib = make_weights(calib_part)

    cv_folds = _build_cv_folds(X_train, y_train, w_train, n_folds=5)
    log.info("training.start", sport=sport, n_train=len(train_part), n_calib=len(calib_part),
             n_hold=len(holdout_df), n_cv_folds=len(cv_folds), n_trials=n_optuna_trials)

    # ── XGBoost Optuna search ─────────────────────────────────────────────────
    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }
        fold_losses = []
        for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
            clf = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
            )
            clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            proba = clf.predict_proba(X_val)[:, 1]
            fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
        return float(np.mean(fold_losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    xgb_study = optuna.create_study(direction="minimize", pruner=pruner)
    xgb_study.optimize(xgb_objective, n_trials=n_optuna_trials, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    log.info("optuna.xgb_best", sport=sport, params=best_xgb_params, cv_loss=xgb_study.best_value)

    # Refine n_estimators with early stopping against calib set
    _finder = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=2000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42,
    )
    _finder.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_calib, y_calib)],
        sample_weight_eval_set=[w_calib],
        verbose=False,
    )
    best_n_trees = _finder.best_iteration or best_xgb_params["n_estimators"]
    log.info("xgb.early_stopping", sport=sport, best_n_trees=best_n_trees)

    # Final XGBoost on train_part with optimal tree count
    best_xgb = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=best_n_trees,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    best_xgb.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    # ── Optional LightGBM search ──────────────────────────────────────────────
    lgb_clf = None
    best_lgb_params: dict = {}

    if _HAS_LGB:
        def lgb_objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            fold_losses = []
            for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
                clf = lgb.LGBMClassifier(
                    **params,
                    objective="binary",
                    random_state=42,
                    verbose=-1,
                )
                clf.fit(X_tr, y_tr, sample_weight=w_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(40, verbose=False),
                                   lgb.log_evaluation(-1)])
                proba = clf.predict_proba(X_val)[:, 1]
                fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
            return float(np.mean(fold_losses))

        lgb_study = optuna.create_study(direction="minimize", pruner=pruner)
        lgb_study.optimize(lgb_objective, n_trials=n_optuna_trials, show_progress_bar=True)
        best_lgb_params = lgb_study.best_params
        log.info("optuna.lgb_best", sport=sport, params=best_lgb_params, cv_loss=lgb_study.best_value)

        lgb_clf = lgb.LGBMClassifier(
            **{k: v for k, v in best_lgb_params.items() if k != "n_estimators"},
            n_estimators=2000,
            objective="binary",
            random_state=42,
            verbose=-1,
        )
        lgb_clf.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_calib, y_calib)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        log.info("lgb.best_iteration", sport=sport,
                 n_trees=getattr(lgb_clf, "best_iteration_", "n/a"))

    # ── Ensemble + manual isotonic calibration ────────────────────────────────
    xgb_raw = best_xgb.predict_proba(X_calib)[:, 1]
    if lgb_clf is not None:
        lgb_raw = lgb_clf.predict_proba(X_calib)[:, 1]
        ensemble_raw = 0.6 * xgb_raw + 0.4 * lgb_raw
    else:
        ensemble_raw = xgb_raw

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ensemble_raw, y_calib, sample_weight=w_calib)

    calibrated = IsotonicCalibratedEnsemble(
        xgb_clf=best_xgb,
        iso=iso,
        lgb_clf=lgb_clf,
        lgb_weight=0.4,
    )

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    proba_hold = calibrated.predict_proba(X_hold)[:, 1]
    metrics = compute_all_winner_metrics(y_hold, proba_hold)
    log.info("winner.holdout_metrics", sport=sport, **metrics)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    flat_params: dict = {}
    for k, v in best_xgb_params.items():
        flat_params[f"xgb_{k}"] = v
    flat_params["xgb_best_n_trees"] = best_n_trees
    if lgb_clf is not None:
        for k, v in best_lgb_params.items():
            flat_params[f"lgb_{k}"] = v
    flat_params["ensemble"] = "xgb+lgb" if lgb_clf is not None else "xgb"

    run_id = log_model_run(
        run_name=run_name or f"{sport}_winner_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="winner",
        target="home_won",
        model=calibrated,
        metrics=metrics,
        params=flat_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["game_date"].min()),
            str(training_df["game_date"].max()),
        ),
        model_framework="sklearn",
    )

    return run_id, metrics


def should_promote(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    min_logloss_improvement: float = 0.005,
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
