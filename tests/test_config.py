from __future__ import annotations

from pathlib import Path

import pytest

from fashion_mnist_mlop.config import load_config


def test_config_loads_toml_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "training.toml"
    config_file.write_text(
        """
[training]
epochs = 3
batch_size = 32
learning_rate = 0.002
optimizer = \"sgd\"
loss_function = \"cross_entropy\"
random_seed = 7
validation_split = 0.25

[mlflow]
tracking_uri = \"http://localhost:5000\"
experiment_name = \"demo\"

[minio]
endpoint = \"localhost:9000\"
access_key = \"demo-user\"
secret_key = \"demo-pass\"
secure = false
bucket_name = \"fashion-minst-bucket\"
model_prefix = \"model/\"

[webhook]
url = \"http://localhost:8001/webhook\"
enabled = false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file)
    assert config.training.epochs == 3
    assert config.training.batch_size == 32
    assert config.training.learning_rate == 0.002
    assert config.training.optimizer == "sgd"
    assert config.mlflow.experiment_name == "demo"
    assert config.minio.bucket_name == "fashion-minst-bucket"
    assert config.webhook.enabled is False


def test_config_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("TRAIN_EPOCHS", "9")
    monkeypatch.setenv("TRAIN_BATCH_SIZE", "128")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "override-user")

    config = load_config()
    assert config.training.epochs == 9
    assert config.training.batch_size == 128
    assert config.minio.access_key == "override-user"


def test_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file does not exist"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("epochs", 0, "epochs"),
        ("batch_size", 0, "batch size"),
        ("learning_rate", 0, "learning rate"),
        ("optimizer", "unsupported", "optimizer"),
        ("loss_function", "unsupported", "loss function"),
        ("validation_split", 1.0, "validation split"),
        ("learning_rate_schedule", "unsupported", "schedule"),
    ],
)
def test_config_rejects_invalid_training_settings(tmp_path: Path, setting: str, value, message: str) -> None:
    config_file = tmp_path / "training.toml"
    config_file.write_text(f"[training]\n{setting} = {value!r}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_file)


def test_config_rejects_invalid_boolean(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("WEBHOOK_ENABLED", raising=False)
    config_file = tmp_path / "training.toml"
    config_file.write_text("[webhook]\nenabled = \"sometimes\"\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid boolean value"):
        load_config(config_file)
