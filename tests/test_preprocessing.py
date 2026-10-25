from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from fashion_mnist_mlop.data import preprocess_image_bytes


def test_preprocess_image_bytes_produces_expected_shape() -> None:
    image_data = np.zeros((28, 28), dtype=np.uint8)
    image = Image.fromarray(image_data, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    tensor = preprocess_image_bytes(payload)
    assert tensor.shape == (1, 1, 28, 28)
    assert str(tensor.dtype) == "torch.float32"
