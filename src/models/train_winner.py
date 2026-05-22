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

import os
import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.eval.metrics import compute_all_winner_metrics
from src.models.registry import log_model_run

log = get_logger(__name__)

_LAMBDA = {"nba": 0.30, "mlb": 0.20}

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# Use GPU if available
try:
    import subprocess

    _GPU = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
except Exception:
    _GPU = False

_XGB_DEVICE = "cuda" if _GPU else "cpu"
_LGB_DEVICE = "cpu"  # LightGBM GPU requires OpenCL, not CUDA; CPU is fast enough


def _optuna_callback(model_tag: str, n_trials: int, start_time: float):
    """Optuna callback: prints one line per trial."""

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed = time.time() - start_time
        is_best = trial.value == study.best_value
        marker = " ◀ best" if is_best else ""
        print(
            f"  [{model_tag}] trial {trial.number + 1:>4}/{n_trials}"
            f"  loss={trial.value:.5f}"
            f"  best={study.best_value:.5f}"
            f"  elapsed={elapsed:.0f}s"
            f"{marker}",
            flush=True,
        )

    return _cb


class _XGBProgressCallback(xgb.callback.TrainingCallback):
    """Prints XGBoost round progress every N rounds during early-stopping fit."""

    def __init__(self, every: int = 100, tag: str = "XGB") -> None:
        self.every = every
        self.tag = tag
        self._start = time.time()

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if (epoch + 1) % self.every == 0:
            val_loss = None
            for ds_metrics in evals_log.values():
                if "logloss" in ds_metrics:
                    val_loss = ds_metrics["logloss"][-1]
            elapsed = time.time() - self._start
            print(
                f"  [{self.tag}] round {epoch + 1:>5}  val_loss={val_loss:.5f}"
                if val_loss
                else f"  [{self.tag}] round {epoch + 1:>5}",
                f"  elapsed={elapsed:.0f}s",
                flush=True,
            )
        return False


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


def _build_cv_folds(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5) -> list[tuple]:
    """Build expanding-window time-series CV folds (data must be sorted by date)."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        tr_end = i * fold_size
        val_end = (i + 1) * fold_size
        if val_end > n:
            break
        folds.append(
            (
                X[:tr_end],
                y[:tr_end],
                w[:tr_end],
                X[tr_end:val_end],
                y[tr_end:val_end],
                w[tr_end:val_end],
            )
        )
    return folds


def train_winner_model(
    sport: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 500,
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
    log.info(
        "training.start",
        sport=sport,
        n_train=len(train_part),
        n_calib=len(calib_part),
        n_hold=len(holdout_df),
        n_cv_folds=len(cv_folds),
        n_trials=n_optuna_trials,
    )

    print(f"\n{'=' * 65}")
    print(f"  Training {sport.upper()} winner model")
    print(f"  Train: {len(train_part):,}  Calib: {len(calib_part):,}  Holdout: {len(holdout_df):,}")
    print(
        f"  CV folds: {len(cv_folds)}  Optuna trials: {n_optuna_trials}  Device: {_XGB_DEVICE.upper()}"
    )
    print(f"{'=' * 65}\n")

    # ── XGBoost Optuna search ─────────────────────────────────────────────────
    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
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
                device=_XGB_DEVICE,
                random_state=42,
            )
            clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            proba = clf.predict_proba(X_val)[:, 1]
            fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
        return float(np.mean(fold_losses))

    # ── Load saved params or run Optuna search ───────────────────────────────
    import json

    _params_path = f"reports/{sport}_winner_xgb_params.json"
    os.makedirs("reports", exist_ok=True)

    if n_optuna_trials == 0 and os.path.exists(_params_path):
        with open(_params_path) as _f:
            best_xgb_params = json.load(_f)
        print(f"[XGBoost] Skipping search — loaded saved params from {_params_path}")
        print(f"[XGBoost] Params: {best_xgb_params}\n")
        log.info("optuna.xgb_loaded", sport=sport, params=best_xgb_params)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        xgb_study = optuna.create_study(direction="minimize", pruner=pruner)
        _actual_trials = n_optuna_trials if n_optuna_trials > 0 else 50
        print(
            f"[XGBoost] Starting Optuna search — {_actual_trials} trials, 5-fold walk-forward CV\n"
        )
        _xgb_t0 = time.time()
        xgb_study.optimize(
            xgb_objective,
            n_trials=_actual_trials,
            show_progress_bar=False,
            callbacks=[_optuna_callback("XGB", _actual_trials, _xgb_t0)],
        )
        best_xgb_params = xgb_study.best_params
        print(f"\n[XGBoost] Search done in {time.time() - _xgb_t0:.0f}s")
        print(f"[XGBoost] Best trial: loss={xgb_study.best_value:.5f}  params={best_xgb_params}\n")
        log.info(
            "optuna.xgb_best",
            sport=sport,
            params=best_xgb_params,
            cv_loss=xgb_study.best_value,
            device=_XGB_DEVICE,
        )
        # Save for future fast-resume runs
        with open(_params_path, "w") as _f:
            json.dump(best_xgb_params, _f, indent=2)
        print(f"[XGBoost] Saved best params to {_params_path}\n")

    # Refine n_estimators with early stopping against calib set
    _finder = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=5000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=_XGB_DEVICE,
        early_stopping_rounds=50,
        random_state=42,
        callbacks=[_XGBProgressCallback(every=200, tag="XGB early-stop")],
    )
    print("[XGBoost] Early-stopping fit (max 5000 trees, stop after 50 no-improve) ...")
    _es_t0 = time.time()
    _finder.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_calib, y_calib)],
        sample_weight_eval_set=[w_calib],
        verbose=False,
    )
    best_n_trees = _finder.best_iteration or best_xgb_params["n_estimators"]
    print(f"[XGBoost] Early stop: best_n_trees={best_n_trees}  ({time.time() - _es_t0:.0f}s)\n")
    log.info("xgb.early_stopping", sport=sport, best_n_trees=best_n_trees)

    # Final XGBoost on train_part with optimal tree count
    best_xgb = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=best_n_trees,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=_XGB_DEVICE,
        random_state=42,
    )
    print(f"[XGBoost] Final fit with {best_n_trees} trees ...")
    _f_t0 = time.time()
    best_xgb.fit(X_train, y_train, sample_weight=w_train, verbose=False)
    print(f"[XGBoost] Final fit done ({time.time() - _f_t0:.0f}s)\n")

    # ── Optional LightGBM search ──────────────────────────────────────────────
    lgb_clf = None
    best_lgb_params: dict = {}

    if _HAS_LGB:

        def lgb_objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
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
                    device=_LGB_DEVICE,
                    random_state=42,
                    verbose=-1,
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
                )
                proba = clf.predict_proba(X_val)[:, 1]
                fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
            return float(np.mean(fold_losses))

        _lgb_params_path = f"reports/{sport}_winner_lgb_params.json"
        _lgb_actual_trials = n_optuna_trials if n_optuna_trials > 0 else 0

        if _lgb_actual_trials == 0 and os.path.exists(_lgb_params_path):
            with open(_lgb_params_path) as _f:
                best_lgb_params = json.load(_f)
            print(f"[LightGBM] Skipping search — loaded saved params from {_lgb_params_path}")
            log.info("optuna.lgb_loaded", sport=sport, params=best_lgb_params)
        else:
            _lgb_actual_trials = _lgb_actual_trials or 200
            _lgb_pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
            lgb_study = optuna.create_study(direction="minimize", pruner=_lgb_pruner)
            print(
                f"[LightGBM] Starting Optuna search — {_lgb_actual_trials} trials, 5-fold walk-forward CV\n"
            )
            _lgb_t0 = time.time()
            lgb_study.optimize(
                lgb_objective,
                n_trials=_lgb_actual_trials,
                show_progress_bar=False,
                callbacks=[_optuna_callback("LGB", _lgb_actual_trials, _lgb_t0)],
            )
            best_lgb_params = lgb_study.best_params
            print(f"\n[LightGBM] Search done in {time.time() - _lgb_t0:.0f}s")
            print(
                f"[LightGBM] Best trial: loss={lgb_study.best_value:.5f}  params={best_lgb_params}\n"
            )
            log.info(
                "optuna.lgb_best", sport=sport, params=best_lgb_params, cv_loss=lgb_study.best_value
            )
            with open(_lgb_params_path, "w") as _f:
                json.dump(best_lgb_params, _f, indent=2)
            print(f"[LightGBM] Saved best params to {_lgb_params_path}\n")

        lgb_clf = lgb.LGBMClassifier(
            **{k: v for k, v in best_lgb_params.items() if k != "n_estimators"},
            n_estimators=5000,
            objective="binary",
            device=_LGB_DEVICE,
            random_state=42,
            verbose=-1,
        )
        print("[LightGBM] Final fit with early stopping (max 5000 trees) ...")
        _lgbf_t0 = time.time()
        lgb_clf.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_calib, y_calib)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        lgb_best_iter = getattr(lgb_clf, "best_iteration_", "n/a")
        print(
            f"[LightGBM] Final fit done — best_n_trees={lgb_best_iter}  ({time.time() - _lgbf_t0:.0f}s)\n"
        )
        log.info("lgb.best_iteration", sport=sport, n_trees=lgb_best_iter)

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
    print(f"[Eval] Scoring holdout ({len(y_hold):,} games) ...")
    proba_hold = calibrated.predict_proba(X_hold)[:, 1]
    metrics = compute_all_winner_metrics(y_hold, proba_hold)
    print(f"\n{'=' * 65}")
    print(f"  {sport.upper()} Holdout Results")
    print(f"  Accuracy : {metrics.get('accuracy', 0):.4f}")
    print(f"  Log-loss : {metrics.get('logloss', 0):.5f}")
    print(f"  Brier    : {metrics.get('brier', 0):.5f}")
    print(f"  ECE      : {metrics.get('ece', 0):.5f}")
    print(f"  N samples: {metrics.get('n_samples', len(y_hold)):,}")
    print(f"  Ensemble : {'XGB+LGB' if lgb_clf is not None else 'XGB only'}")
    print(f"{'=' * 65}\n")
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
        return (
            False,
            f"log-loss {chall_ll:.4f} not better than champion {champ_ll:.4f} by {min_logloss_improvement}",
        )
    if chall_ece > champ_ece + max_ece_increase:
        return False, f"ECE {chall_ece:.4f} exceeds champion {champ_ece:.4f} by margin"
    if chall_brier > champ_brier:
        return False, f"Brier {chall_brier:.4f} worse than champion {champ_brier:.4f}"
    return True, "all gates passed"
