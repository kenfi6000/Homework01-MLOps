"""Load a selected model and run Fashion-MNIST image predictions."""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch

from fashion_mnist_mlop.data import CLASS_NAMES, preprocess_image_for_prediction
from fashion_mnist_mlop.minio_utils import MinioArtifactStore
from fashion_mnist_mlop.model import LeNetFashionMNIST


class InferenceModel:
    """Manage model loading and prediction for the API process."""

    def __init__(self) -> None:
        self.model = None
        self.model_key = None
        self.store = MinioArtifactStore.from_env()

    def load_model(self) -> str:
        """Load a local override or the selected/latest version from MinIO."""
        local_model_path = os.getenv("MODEL_PATH")
        if local_model_path:
            path = Path(local_model_path)
            if not path.is_file():
                raise FileNotFoundError(f"Local model file does not exist: {path}")
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            self.model = LeNetFashionMNIST()
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_key = str(path)
            return self.model_key

        requested_key = os.getenv("MODEL_OBJECT_KEY")
        if requested_key:
            candidate = requested_key
        else:
            # Choosing the latest version makes newly published runs available on restart.
            candidate = self.store.latest_object_key(prefix=os.getenv("MODEL_PREFIX", "model/"))

        if not candidate:
            raise RuntimeError("No model is available in MinIO yet. Train a model first or set MODEL_OBJECT_KEY.")

        payload = self.store.download_bytes(candidate)
        state_dict = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        self.model = LeNetFashionMNIST()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_key = candidate
        return candidate

    def predict_image_bytes(self, file_bytes: bytes) -> dict[str, object]:
        """Preprocess image bytes and return the predicted label and probabilities."""
        image_tensor = preprocess_image_for_prediction(file_bytes)
        if self.model is None:
            raise RuntimeError("The model has not been loaded yet.")
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            predicted_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_index].item())
        return {
            "label": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "probabilities": [float(value) for value in probabilities.tolist()],
            "predicted_index": predicted_index,
            "model_key": self.model_key,
        }


def create_inference_model() -> InferenceModel:
    """Create the inference runtime used by the API application."""
    return InferenceModel()
