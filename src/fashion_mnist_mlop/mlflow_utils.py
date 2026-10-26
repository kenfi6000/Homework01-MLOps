"""Helpers for MLflow experiment setup and run metadata artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow


def init_mlflow_experiment(tracking_uri: str, experiment_name: str):
    """Select the tracking server and active experiment for subsequent logging."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(experiment_name)


def log_training_run_metadata(run_id: str, config: dict, validation_accuracy: float, model_key: str) -> None:
    """Write non-secret run details to a structured MLflow artifact."""
    payload = {
        "run_id": run_id,
        "model_bucket": config["minio"]["bucket_name"],
        "model_object_key": model_key,
        "validation_accuracy": validation_accuracy,
        "training": config["training"],
        "mlflow": config["mlflow"],
        "minio": config["minio"],
        "webhook": config["webhook"],
    }
    with TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / "run_metadata.json"
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifact_path), artifact_path="metadata")
