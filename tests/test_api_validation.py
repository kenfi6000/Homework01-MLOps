"""Tests for API health and validation of uploaded image requests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fashion_mnist_mlop.api import app

client = TestClient(app)


def test_health_endpoint_returns_json() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload


def test_predict_rejects_non_image_upload() -> None:
    data = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    response = client.post("/predict", files=data)
    assert response.status_code == 400
    assert "Only image files are accepted" in response.json()["detail"]


def test_predict_rejects_empty_image_upload() -> None:
    data = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/predict", files=data)
    assert response.status_code == 400
    assert "Uploaded image is empty" in response.json()["detail"]


def test_predict_rejects_invalid_image_payload() -> None:
    data = {"file": ("invalid.png", b"not a png image", "image/png")}
    response = client.post("/predict", files=data)
    assert response.status_code == 400
    assert "Invalid image payload" in response.json()["detail"]
