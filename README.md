# Fashion-MNIST MLOps Pipeline

An end-to-end application for training a LeNet-style CNN on Fashion-MNIST, tracking experiments in MLflow, storing versioned models in MinIO, and serving predictions through FastAPI and Streamlit.

## Architecture and classes

```text
Fashion-MNIST -> PyTorch training -> MLflow parameters, metrics, and metadata
                                  -> fashion-minst-bucket/model/<run-id>/model.pt
                                  -> webhook after the model is published
Streamlit -> POST /predict -> FastAPI -> model downloaded from MinIO
```

The bucket name `fashion-minst-bucket` intentionally preserves the requested spelling (`minst`).
The ten classes are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## Prerequisites

- Python 3.11 or later
- Docker Desktop with Docker Compose v2 for the complete environment
- Internet access during the first training run to download Fashion-MNIST

The `minioadmin` fallback credentials are for local development only. Configure your own local credentials in `.env` before starting the services, and do not reuse these defaults in production.

## Local installation

From PowerShell at the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

If PowerShell blocks `Activate.ps1`, you do not need to activate the environment. Run `.venv\Scripts\python.exe` and `.venv\Scripts\pytest.exe` directly.

Update `.env` before starting services. Git ignores `.env`; `.env.example` contains example values only.

## Training configuration

Default settings are in [`configs/training.toml`](configs/training.toml). Each setting can be overridden by its corresponding `TRAIN_*` environment variable or a supported command-line option.

| TOML setting | Environment variable | CLI | Default |
| --- | --- | --- | --- |
| `epochs` | `TRAIN_EPOCHS` | `--epochs` | 5 |
| `batch_size` | `TRAIN_BATCH_SIZE` | `--batch-size` | 64 |
| `learning_rate` | `TRAIN_LEARNING_RATE` | `--learning-rate` | 0.001 |
| `optimizer` | `TRAIN_OPTIMIZER` | `--optimizer` | `adam` |
| `loss_function` | `TRAIN_LOSS_FUNCTION` | `--loss` | `cross_entropy` |
| `random_seed` | `TRAIN_RANDOM_SEED` | `--seed` | 42 |
| `validation_split` | `TRAIN_VALIDATION_SPLIT` | `--validation-split` | 0.2 |
| `momentum` | `TRAIN_MOMENTUM` | — | 0.9 |
| `weight_decay` | `TRAIN_WEIGHT_DECAY` | — | 0.0001 |
| `learning_rate_schedule` | `TRAIN_LR_SCHEDULE` | — | `none` |
| `early_stopping_patience` | `TRAIN_EARLY_STOPPING_PATIENCE` | — | 5 |
| `data_loader_workers` | `TRAIN_DATA_LOADER_WORKERS` | — | 2 |
| `pin_memory` | `TRAIN_PIN_MEMORY` | — | `true` |

Supported optimizers: `adam`, `sgd`. Supported loss functions: `cross_entropy`, `nll_loss`. Supported schedulers: `none`, `step`. Invalid values are rejected before training starts. The checkpoint with the best validation loss is published.

Examples:

```powershell
.\.venv\Scripts\python.exe -m fashion_mnist_mlop.train --config configs/training.toml
.\.venv\Scripts\python.exe -m fashion_mnist_mlop.train --epochs 8 --batch-size 128 --learning-rate 0.0005 --optimizer adam --loss cross_entropy --seed 42 --validation-split 0.2
```

Each run logs its effective configuration, per-epoch training and validation losses, validation accuracy, best epoch, and a reference to the MinIO model in MLflow. Sensitive credentials and URLs are excluded from configuration artifacts.

## Start the complete stack with Docker Compose

From the repository root, create the local environment file:

```powershell
Copy-Item .env.example .env
```

Set your own local values for `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` in `.env`, then build the images once:

```powershell
docker compose build
```

Start MinIO, MLflow, and the webhook receiver:

```powershell
docker compose up -d minio mlflow webhook-receiver
docker compose ps
```

Run training as a one-shot Compose service. Compose waits for MinIO, MLflow, and the webhook receiver to become healthy:

```powershell
docker compose up train-model
```

When training completes, the terminal displays the MLflow run ID, validation accuracy, and published object key. Then start the API and Streamlit without rebuilding the images or rerunning training:

```powershell
docker compose up -d api streamlit
docker compose ps
```

The application images install CPU-only PyTorch and TorchVision distributions to avoid including CUDA libraries in the containers.

Local service URLs:

| Service | URL |
| --- | --- |
| Streamlit | <http://127.0.0.1:8501> |
| FastAPI | <http://127.0.0.1:8000> |
| OpenAPI documentation | <http://127.0.0.1:8000/docs> |
| MLflow | <http://127.0.0.1:5000> |
| MinIO console | <http://127.0.0.1:9001> |
| Webhook receiver | <http://127.0.0.1:8001/health> |

Compose binds published ports to the local machine only. The development setup uses an archived Bitnami MinIO image that is no longer maintained. For long-term or production use, choose a supported MinIO distribution and configure credentials, TLS, storage, and network access for your environment.

MinIO and MLflow data persist in the Compose volumes `minio_storage` and `mlflow_data`. The API starts only after the training service succeeds; Streamlit calls the API over the Compose network. View service logs with `docker compose logs -f train-model api streamlit`.

## MLflow

Open the MLflow UI at <http://127.0.0.1:5000> and select the `fashion-mnist-classifier` experiment to view parameters, per-epoch losses, validation accuracy, `training_config.json`, and metadata pointing to the MinIO model.

The MLflow backend and artifacts are stored in the persistent `mlflow_data` volume.

## MinIO and model selection

Main settings are `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_SECURE`, `MODEL_BUCKET_NAME`, and `MODEL_PREFIX`. The default bucket is `fashion-minst-bucket` and the default prefix is `model/`. The bucket is created on upload if it does not exist. Each training run publishes its best validation checkpoint to `model/<run-id>/model.pt`, keeping previous versions intact.

At startup, the API loads `MODEL_OBJECT_KEY` if it is set; otherwise, it selects the newest MinIO object under `MODEL_PREFIX`. For local testing, `MODEL_PATH` can be set and takes precedence over MinIO.

## Training-complete webhook

The webhook is sent only after training finishes, MLflow logging succeeds, and the model has been uploaded successfully. Configure its URL and enablement with `WEBHOOK_URL` and `WEBHOOK_ENABLED`. The Compose receiver listens at `http://127.0.0.1:8001/webhook`; inspect its logs with `docker compose logs webhook-receiver`.

Example payload:

```json
{
  "status": "success",
  "run_id": "<mlflow-run-id>",
  "bucket": "fashion-minst-bucket",
  "object_key": "model/<run-id>/model.pt",
  "validation_accuracy": 0.87,
  "experiment_name": "fashion-mnist-classifier"
}
```

## API and user interface

- `GET /health` reports whether the API loaded a model.
- `POST /predict` accepts a single multipart image in the `file` field. The response includes the predicted class, confidence, and probabilities for each class.

PowerShell example:

```powershell
curl.exe -F "file=@tests\fashion_mnist_samples\6_shirt_01.png;type=image/png" http://127.0.0.1:8000/predict
```

The API validates empty files, MIME types, and image contents. If MinIO or the model is unavailable, `/health` reports a degraded status and `/predict` explains that the model is not loaded. Streamlit previews the image, sends its bytes to FastAPI, and displays the predicted class and confidence; it does not run inference itself.

The 28 × 28 grayscale examples in [`tests/fashion_mnist_samples/`](tests/fashion_mnist_samples/) come from the official test split and are described in that directory's README. This classifier is intended for Fashion-MNIST-style images, not real-world clothing photographs.

## Tests and local validation

```powershell
.\.venv\Scripts\pytest.exe tests -q
```

Tests cover configuration loading and validation, 28 × 28 preprocessing, invalid API inputs, and local model loading for automated tests.

## Stop the services

Stop the containers without deleting persisted data:

```powershell
docker compose down
```

Only remove persistent volumes with `docker compose down -v` if you intend to delete the MinIO models, MLflow experiments, and associated data.
