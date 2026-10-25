from __future__ import annotations

import argparse
import copy
import os
import random
from collections.abc import Sequence
from io import BytesIO

import mlflow
import numpy as np
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from fashion_mnist_mlop.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    validate_training_settings,
)
from fashion_mnist_mlop.data import build_data_loaders, build_fashion_mnist_data
from fashion_mnist_mlop.minio_utils import MinioArtifactStore
from fashion_mnist_mlop.mlflow_utils import (
    init_mlflow_experiment,
    log_training_run_metadata,
)
from fashion_mnist_mlop.model import build_model


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST classifier and publish it to MinIO.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to TOML configuration file")
    parser.add_argument("--epochs", type=int, help="Override the configured number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size")
    parser.add_argument("--learning-rate", type=float, help="Override the configured learning rate")
    parser.add_argument("--optimizer", type=str, help="Override the configured optimizer")
    parser.add_argument("--loss", "--loss-function", dest="loss", type=str, help="Override the configured loss function")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--validation-split", type=float, help="Override the configured validation split")
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_optimizer(model: nn.Module, training_config: dict, optimizer_name: str) -> torch.optim.Optimizer:
    if optimizer_name.lower() == "sgd":
        return SGD(model.parameters(), lr=training_config["learning_rate"], momentum=training_config["momentum"], weight_decay=training_config["weight_decay"])
    if optimizer_name.lower() == "adam":
        return Adam(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"])
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _calculate_loss(logits: torch.Tensor, targets: torch.Tensor, loss_function: str) -> torch.Tensor:
    if loss_function.lower() == "cross_entropy":
        return F.cross_entropy(logits, targets)
    if loss_function.lower() == "nll_loss":
        return F.nll_loss(F.log_softmax(logits, dim=1), targets)
    raise ValueError(f"Unsupported loss function: {loss_function}")


def _safe_config_snapshot(config) -> dict:
    return {
        "training": config.training.to_dict(),
        "mlflow": {
            "tracking_uri": "<redacted>",
            "experiment_name": config.mlflow.experiment_name,
        },
        "minio": {
            "endpoint": config.minio.endpoint,
            "secure": config.minio.secure,
            "bucket_name": config.minio.bucket_name,
            "model_prefix": config.minio.model_prefix,
        },
        "webhook": {
            "enabled": config.webhook.enabled,
            "url_configured": bool(config.webhook.url),
        },
    }


def _send_webhook_if_needed(url: str, payload: dict) -> None:
    if not url:
        return
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()


def train_and_publish(config_path: str | None = None, args: argparse.Namespace | None = None) -> str:
    config = load_config(config_path)
    args = args or argparse.Namespace(
        epochs=None,
        batch_size=None,
        learning_rate=None,
        optimizer=None,
        loss=None,
        seed=None,
        validation_split=None,
    )
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.optimizer is not None:
        config.training.optimizer = args.optimizer
    if args.loss is not None:
        config.training.loss_function = args.loss
    if args.seed is not None:
        config.training.random_seed = args.seed
    if args.validation_split is not None:
        config.training.validation_split = args.validation_split
    validate_training_settings(config.training)

    training_cfg = config.training.to_dict()
    _set_seed(config.training.random_seed)

    train_dataset, validation_dataset = build_fashion_mnist_data(training_cfg)
    train_loader, validation_loader = build_data_loaders(training_cfg, train_dataset, validation_dataset)

    model = build_model()
    optimizer = _create_optimizer(model, training_cfg, config.training.optimizer)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5) if config.training.learning_rate_schedule.lower() == "step" else None

    init_mlflow_experiment(config.mlflow.tracking_uri, config.mlflow.experiment_name)
    best_state = None
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    completed_epochs = 0

    with mlflow.start_run(run_name=config.mlflow.experiment_name) as mlflow_run:
        mlflow.log_params(
            {
                **training_cfg,
                "model_class": "LeNetFashionMNIST",
            }
        )

        for epoch in range(1, config.training.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                logits = model(inputs)
                loss = _calculate_loss(logits, targets, config.training.loss_function)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            train_loss = epoch_loss / len(train_loader.dataset)

            model.eval()
            validation_loss = 0.0
            validation_correct = 0
            validation_total = 0
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    logits = model(inputs)
                    loss = _calculate_loss(logits, targets, config.training.loss_function)
                    validation_loss += loss.item() * inputs.size(0)
                    predictions = torch.argmax(logits, dim=1)
                    validation_correct += int((predictions == targets).sum().item())
                    validation_total += targets.size(0)
            validation_loss /= len(validation_loader.dataset)
            validation_accuracy = validation_correct / validation_total
            completed_epochs = epoch

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", validation_accuracy, step=epoch)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if scheduler is not None:
                scheduler.step()
            if patience_counter >= config.training.early_stopping_patience:
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a model checkpoint.")

        model.load_state_dict(best_state)
        model_prefix = config.minio.model_prefix.strip("/")
        object_key = f"{model_prefix}/{mlflow_run.info.run_id}/model.pt" if model_prefix else f"{mlflow_run.info.run_id}/model.pt"
        model_buffer = BytesIO()
        torch.save(model.state_dict(), model_buffer)

        minio_store = MinioArtifactStore(
            endpoint=config.minio.endpoint,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=config.minio.secure,
            bucket_name=config.minio.bucket_name,
            model_prefix=config.minio.model_prefix,
        )
        minio_store.upload_bytes(object_key, model_buffer.getvalue())
        mlflow.log_params(
            {
                "model_bucket": config.minio.bucket_name,
                "model_object_key": object_key,
            }
        )

        config_snapshot = _safe_config_snapshot(config)
        mlflow.log_metric("completed_epochs", completed_epochs)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_validation_accuracy", best_validation_accuracy)
        mlflow.log_metric("final_validation_accuracy", best_validation_accuracy)
        mlflow.log_dict(config_snapshot, artifact_file="training_config.json")
        log_training_run_metadata(mlflow_run.info.run_id, config_snapshot, best_validation_accuracy, object_key)
        run_id = mlflow_run.info.run_id

    if config.webhook.enabled and config.webhook.url:
        payload = {
            "status": "success",
            "run_id": run_id,
            "bucket": config.minio.bucket_name,
            "model_bucket": config.minio.bucket_name,
            "object_key": object_key,
            "model_object_key": object_key,
            "validation_accuracy": best_validation_accuracy,
            "experiment_name": config.mlflow.experiment_name,
        }
        _send_webhook_if_needed(config.webhook.url, payload)

    print(f"Training complete. MLflow run: {run_id}")
    print(f"Model uploaded to MinIO: {config.minio.bucket_name}/{object_key}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f} (epoch {best_epoch})")
    return run_id


def main() -> None:
    config_path = os.getenv("TRAIN_CONFIG_PATH")
    args = _parse_args()
    train_and_publish(config_path or args.config, args)


if __name__ == "__main__":
    main()
