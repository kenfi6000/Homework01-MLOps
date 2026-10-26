"""Streamlit presentation layer that delegates predictions to the FastAPI service."""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")


def show_health_status() -> None:
    """Display API availability without blocking the image upload workflow."""
    try:
        response = requests.get(f"{API_URL.rstrip('/')}/health", timeout=10)
        if response.ok:
            payload = response.json()
            st.caption(f"API status: {payload.get('status', 'unknown')} | model loaded: {payload.get('model_loaded', False)}")
        else:
            st.caption("API status: unavailable")
    except requests.RequestException:
        st.caption("API status: unavailable")


st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👗")
st.title("Fashion-MNIST classifier")
show_health_status()

uploaded_file = st.file_uploader("Upload a 28x28 Fashion-MNIST-style image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        # getvalue() can be called after the preview has consumed the upload stream.
        file_data = uploaded_file.getvalue()
        try:
            response = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                files={"file": (uploaded_file.name, file_data, uploaded_file.type or "image/png")},
                timeout=90,
            )
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()

        if response.status_code == 503:
            st.warning("The training model is not available in MinIO yet. Please run the training job first.")
        elif response.status_code >= 400:
            st.error(f"Prediction failed: {response.json().get('detail', response.text)}")
        else:
            payload = response.json()
            st.success(f"Predicted class: {payload['label']}")
            st.metric("Confidence", f"{payload['confidence']:.2%}")
            st.json(payload)
else:
    st.info("Upload a single grayscale image to predict the Fashion-MNIST class.")
