"""Fashion-MNIST loading, image preprocessing, and classification helpers."""

from __future__ import annotations

import io
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def make_transforms() -> transforms.Compose:
    """Apply the tensor conversion and normalization used during training."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,)),
        ]
    )


def build_fashion_mnist_data(config: dict) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Download Fashion-MNIST and create a seeded train/validation split."""
    dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=make_transforms(),
    )
    total = len(dataset)
    validation_count = int(total * config["validation_split"])
    train_count = total - validation_count
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_count, validation_count],
        generator=torch.Generator().manual_seed(config["random_seed"]),
    )
    return train_dataset, validation_dataset


def build_data_loaders(config: dict, train_dataset, validation_dataset) -> tuple[DataLoader, DataLoader]:
    """Create shuffled training and deterministic validation data loaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["data_loader_workers"],
        pin_memory=config.get("pin_memory", True),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["data_loader_workers"],
        pin_memory=config.get("pin_memory", True),
    )
    return train_loader, validation_loader


def preprocess_image_bytes(file_bytes: bytes) -> torch.Tensor:
    """Convert uploaded image bytes into a normalized 1x1x28x28 model input."""
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.286) / 0.353
    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


def preprocess_image_for_prediction(file_bytes: bytes) -> torch.Tensor:
    """Expose the shared preprocessing path used by prediction callers."""
    return preprocess_image_bytes(file_bytes)


def detect_class_name(prediction_index: int) -> str:
    """Map a Fashion-MNIST class index to its display name."""
    return CLASS_NAMES[prediction_index]


def compute_accuracy(predictions: Iterable[int], targets: Iterable[int]) -> float:
    """Calculate accuracy for paired class predictions and targets."""
    correct = 0
    total = 0
    for prediction, target in zip(predictions, targets):
        correct += int(prediction == target)
        total += 1
    if total == 0:
        return 0.0
    return correct / total
