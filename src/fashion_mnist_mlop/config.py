"""Load and validate runtime settings from TOML files and environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "training.toml"


def _read_toml(path: str | Path | None) -> dict[str, Any]:
    """Read the requested TOML file, or return defaults when no default file exists."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.exists():
        if path is not None:
            raise FileNotFoundError(f"Training configuration file does not exist: {resolved}")
        return {}
    with resolved.open("rb") as handle:
        return tomllib.load(handle)


def _read_env(key: str, default: Any, cast: Any | None = None) -> Any:
    """Return an environment override converted to the requested type."""
    value = os.environ.get(key, default)
    if cast is None:
        return value
    return cast(value)


@dataclass
class TrainingSettings:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    random_seed: int = 42
    validation_split: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    learning_rate_schedule: str = "none"
    early_stopping_patience: int = 5
    data_loader_workers: int = 2
    pin_memory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "random_seed": self.random_seed,
            "validation_split": self.validation_split,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "learning_rate_schedule": self.learning_rate_schedule,
            "early_stopping_patience": self.early_stopping_patience,
            "data_loader_workers": self.data_loader_workers,
            "pin_memory": self.pin_memory,
        }


@dataclass
class MLflowSettings:
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "fashion-mnist-classifier"

    def to_dict(self) -> dict[str, Any]:
        return {"tracking_uri": self.tracking_uri, "experiment_name": self.experiment_name}


@dataclass
class MinioSettings:
    endpoint: str = "minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    bucket_name: str = "fashion-minst-bucket"
    model_prefix: str = "model/"

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "secure": self.secure,
            "bucket_name": self.bucket_name,
            "model_prefix": self.model_prefix,
        }


@dataclass
class WebhookSettings:
    url: str = "http://webhook-receiver:8001/webhook"
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "enabled": self.enabled}


@dataclass
class RuntimeConfig:
    training: TrainingSettings
    mlflow: MLflowSettings
    minio: MinioSettings
    webhook: WebhookSettings


def _load_training_settings(raw: dict[str, Any]) -> TrainingSettings:
    """Resolve training options with environment values taking precedence over TOML."""
    section = raw.get("training", {})
    return TrainingSettings(
        epochs=_read_env("TRAIN_EPOCHS", section.get("epochs", 5), int),
        batch_size=_read_env("TRAIN_BATCH_SIZE", section.get("batch_size", 64), int),
        learning_rate=_read_env("TRAIN_LEARNING_RATE", section.get("learning_rate", 1e-3), float),
        optimizer=_read_env("TRAIN_OPTIMIZER", section.get("optimizer", "adam"), str),
        loss_function=_read_env("TRAIN_LOSS_FUNCTION", section.get("loss_function", "cross_entropy"), str),
        random_seed=_read_env("TRAIN_RANDOM_SEED", section.get("random_seed", 42), int),
        validation_split=_read_env("TRAIN_VALIDATION_SPLIT", section.get("validation_split", 0.2), float),
        momentum=_read_env("TRAIN_MOMENTUM", section.get("momentum", 0.9), float),
        weight_decay=_read_env("TRAIN_WEIGHT_DECAY", section.get("weight_decay", 1e-4), float),
        learning_rate_schedule=_read_env("TRAIN_LR_SCHEDULE", section.get("learning_rate_schedule", "none"), str),
        early_stopping_patience=_read_env(
            "TRAIN_EARLY_STOPPING_PATIENCE",
            section.get("early_stopping_patience", 5),
            int,
        ),
        data_loader_workers=_read_env("TRAIN_DATA_LOADER_WORKERS", section.get("data_loader_workers", 2), int),
        pin_memory=_read_env("TRAIN_PIN_MEMORY", section.get("pin_memory", True), _as_bool),
    )


def _load_mlflow_settings(raw: dict[str, Any]) -> MLflowSettings:
    """Resolve MLflow connection settings from the environment or TOML."""
    section = raw.get("mlflow", {})
    return MLflowSettings(
        tracking_uri=_read_env("MLFLOW_TRACKING_URI", section.get("tracking_uri", "http://mlflow:5000"), str),
        experiment_name=_read_env("MLFLOW_EXPERIMENT_NAME", section.get("experiment_name", "fashion-mnist-classifier"), str),
    )


def _load_minio_settings(raw: dict[str, Any]) -> MinioSettings:
    """Resolve MinIO settings without including credentials in serialized config."""
    section = raw.get("minio", {})
    return MinioSettings(
        endpoint=_read_env("MINIO_ENDPOINT", section.get("endpoint", "minio:9000"), str),
        access_key=_read_env("MINIO_ACCESS_KEY", section.get("access_key", "minioadmin"), str),
        secret_key=_read_env("MINIO_SECRET_KEY", section.get("secret_key", "minioadmin"), str),
        secure=_read_env("MINIO_SECURE", section.get("secure", False), _as_bool),
        bucket_name=_read_env("MODEL_BUCKET_NAME", section.get("bucket_name", "fashion-minst-bucket"), str),
        model_prefix=_read_env("MODEL_PREFIX", section.get("model_prefix", "model/"), str),
    )


def _load_webhook_settings(raw: dict[str, Any]) -> WebhookSettings:
    """Resolve webhook settings from the environment or TOML."""
    section = raw.get("webhook", {})
    return WebhookSettings(
        url=_read_env("WEBHOOK_URL", section.get("url", "http://webhook-receiver:8001/webhook"), str),
        enabled=_read_env("WEBHOOK_ENABLED", section.get("enabled", True), _as_bool),
    )


def _as_bool(value: str | bool) -> bool:
    """Parse common human-readable boolean values and reject ambiguous strings."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_config(config_path: str | Path | None = None) -> RuntimeConfig:
    """Load every settings group and validate training options before use."""
    raw = _read_toml(config_path or DEFAULT_CONFIG_PATH)
    training = _load_training_settings(raw)
    validate_training_settings(training)
    mlflow = _load_mlflow_settings(raw)
    minio = _load_minio_settings(raw)
    webhook = _load_webhook_settings(raw)
    return RuntimeConfig(training=training, mlflow=mlflow, minio=minio, webhook=webhook)


def validate_training_settings(settings: TrainingSettings) -> None:
    """Reject unsupported or unsafe training values before starting a run."""
    if settings.epochs < 1:
        raise ValueError("Training epochs must be at least 1.")
    if settings.batch_size < 1:
        raise ValueError("Training batch size must be at least 1.")
    if not isfinite(settings.learning_rate) or settings.learning_rate <= 0:
        raise ValueError("Training learning rate must be greater than 0.")
    if settings.optimizer.lower() not in {"adam", "sgd"}:
        raise ValueError("Training optimizer must be 'adam' or 'sgd'.")
    if settings.loss_function.lower() not in {"cross_entropy", "nll_loss"}:
        raise ValueError("Training loss function must be 'cross_entropy' or 'nll_loss'.")
    if not 0 < settings.validation_split < 1:
        raise ValueError("Training validation split must be between 0 and 1.")
    if not isfinite(settings.momentum) or settings.momentum < 0:
        raise ValueError("Training momentum must not be negative.")
    if not isfinite(settings.weight_decay) or settings.weight_decay < 0:
        raise ValueError("Training weight decay must not be negative.")
    if settings.learning_rate_schedule.lower() not in {"none", "step"}:
        raise ValueError("Training learning rate schedule must be 'none' or 'step'.")
    if settings.early_stopping_patience < 1:
        raise ValueError("Training early stopping patience must be at least 1.")
    if settings.data_loader_workers < 0:
        raise ValueError("Training data loader workers must not be negative.")
