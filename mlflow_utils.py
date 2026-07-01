import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def init_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "customer-churn-experiments") -> str:
    if not HAS_MLFLOW:
        logger.warning("MLflow not installed; skipping MLflow initialization.")
        return "mlruns"

    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning("Unable to initialize MLflow: %s", e)
    return tracking_uri


@contextmanager
def start_run(run_name: Optional[str] = None, nested: bool = False):
    if HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name=run_name, nested=nested):
                yield mlflow.active_run()
        except Exception as e:
            logger.warning("Unable to start MLflow run: %s", e)
            yield None
    else:
        yield None


def mlflow_is_active() -> bool:
    if not HAS_MLFLOW:
        return False
    try:
        return mlflow.active_run() is not None
    except Exception:
        return False


def log_params(params: Dict[str, Any]) -> None:
    if HAS_MLFLOW and params:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: Dict[str, float]) -> None:
    if HAS_MLFLOW and metrics:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def log_artifact(path: Path, artifact_path: Optional[str] = None) -> None:
    if HAS_MLFLOW and path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_model(model: Any, artifact_path: str = "model", registered_model_name: Optional[str] = None, sample_input: Optional[Any] = None) -> None:
    if not HAS_MLFLOW:
        return
    signature = None
    if sample_input is not None:
        try:
            signature = infer_signature(sample_input, model.predict(sample_input))
        except Exception:
            signature = None
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        signature=signature,
    )
