"""MLflow model registry wrapper.

MLflow is the source of truth for versioned artifacts.
The `models` DB table is a mirror for SQL joins on predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.db.models import ModelRecord

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_model_run(
    run_name: str,
    sport: str,
    kind: str,
    target: str,
    model: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    feature_names: list[str],
    training_range: tuple[str, str],
    model_framework: str = "sklearn",
) -> str:
    """Log a training run to MLflow. Returns the run_id."""
    _setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "sport": sport,
                "kind": kind,
                "target": target,
                "training_start": training_range[0],
                "training_end": training_range[1],
                "n_features": len(feature_names),
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        if model_framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_framework == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        log.info(
            "mlflow.run_logged",
            run_id=run.info.run_id,
            sport=sport,
            kind=kind,
            target=target,
            metrics=metrics,
        )
        return run.info.run_id


def load_model(run_id: str, framework: str = "sklearn") -> Any:
    """Load a model artifact from MLflow by run_id."""
    _setup_mlflow()
    uri = f"runs:/{run_id}/model"
    if framework == "xgboost":
        return mlflow.xgboost.load_model(uri)
    elif framework == "lightgbm":
        return mlflow.lightgbm.load_model(uri)
    return mlflow.sklearn.load_model(uri)


def get_run_metrics(run_id: str) -> dict[str, float]:
    _setup_mlflow()
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def promote_model(
    session: Any,  # SQLAlchemy Session
    run_id: str,
    sport_id: int,
    kind: str,
    target: str,
    version: str,
    metrics: dict[str, float],
    feature_spec_hash: str,
) -> ModelRecord:  # type: ignore[name-defined]
    """Deactivate old active model for this (sport, kind, target), activate new one."""
    from src.core.time import utc_now
    from src.db.models import ModelRecord

    # Deactivate existing
    session.query(ModelRecord).filter_by(
        sport_id=sport_id, kind=kind, target=target, active=True
    ).update({"active": False})

    record = ModelRecord(
        sport_id=sport_id,
        kind=kind,
        target=target,
        version=version,
        mlflow_run_id=run_id,
        trained_at=utc_now(),
        active=True,
        metrics=metrics,
        feature_spec_hash=feature_spec_hash,
    )
    session.add(record)
    session.flush()
    log.info("model.promoted", sport_id=sport_id, kind=kind, target=target, version=version)
    return record


def rollback_model(session: Any, sport_id: int, kind: str, target: str) -> bool:
    """Revert to the previous active model version."""
    from src.db.models import ModelRecord

    models = (
        session.query(ModelRecord)
        .filter_by(sport_id=sport_id, kind=kind, target=target)
        .order_by(ModelRecord.trained_at.desc())
        .limit(2)
        .all()
    )

    if len(models) < 2:
        log.warning("model.rollback_impossible", sport_id=sport_id, kind=kind, target=target)
        return False

    current, previous = models[0], models[1]
    current.active = False
    previous.active = True
    session.flush()
    log.info("model.rolled_back", to_version=previous.version)
    return True
