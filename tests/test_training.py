"""Tests for configurable training helpers and safe MLflow metadata."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fashion_mnist_mlop.config import load_config
from fashion_mnist_mlop.train import (
    _calculate_loss,
    _create_optimizer,
    _parse_args,
    _safe_config_snapshot,
)


@pytest.mark.parametrize("loss_function", ["cross_entropy", "nll_loss"])
def test_calculate_loss_supports_configured_loss_functions(loss_function: str) -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    targets = torch.tensor([0, 1])

    loss = _calculate_loss(logits, targets, loss_function)

    assert loss.item() > 0


def test_calculate_loss_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError, match="Unsupported loss function"):
        _calculate_loss(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), "unknown")


def test_create_optimizer_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        _create_optimizer(nn.Linear(2, 2), {"learning_rate": 0.1}, "unknown")


def test_parse_args_supports_reproducible_training_overrides() -> None:
    args = _parse_args(["--epochs", "3", "--loss-function", "nll_loss", "--validation-split", "0.25"])

    assert args.epochs == 3
    assert args.loss == "nll_loss"
    assert args.validation_split == 0.25


def test_mlflow_config_snapshot_excludes_credentials(monkeypatch) -> None:
    monkeypatch.setenv("MINIO_ACCESS_KEY", "private-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "private-secret-key")
    config = load_config()

    snapshot = _safe_config_snapshot(config)

    assert "access_key" not in snapshot["minio"]
    assert "secret_key" not in snapshot["minio"]
    assert snapshot["mlflow"]["tracking_uri"] == "<redacted>"
    assert "private-access-key" not in str(snapshot)
    assert "private-secret-key" not in str(snapshot)
