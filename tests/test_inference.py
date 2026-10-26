"""Tests for loading model weights from a local path during inference."""

from __future__ import annotations

import pytest
import torch

from fashion_mnist_mlop.inference import InferenceModel
from fashion_mnist_mlop.model import LeNetFashionMNIST


def test_load_model_from_local_file(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.pt"
    torch.save(LeNetFashionMNIST().state_dict(), model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    inference_model = InferenceModel()
    loaded_path = inference_model.load_model()

    assert loaded_path == str(model_path)
    assert inference_model.model is not None
    assert not inference_model.model.training


def test_load_model_reports_missing_local_file(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "missing.pt"
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with pytest.raises(FileNotFoundError, match="Local model file does not exist"):
        InferenceModel().load_model()
