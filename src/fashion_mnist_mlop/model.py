"""LeNet-style convolutional network for 28x28 grayscale Fashion-MNIST images."""

from __future__ import annotations

from torch import nn


class LeNetFashionMNIST(nn.Module):
    """Extract image features with convolutions and classify them into ten classes."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs):
        """Return class logits for a batch of normalized grayscale images."""
        features = self.feature_extractor(inputs)
        return self.classifier(features)


def build_model() -> LeNetFashionMNIST:
    """Construct the default Fashion-MNIST classifier."""
    return LeNetFashionMNIST()
