from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from fashion_mnist_mlop.inference import create_inference_model

inference_runtime = create_inference_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        inference_runtime.load_model()
        app.state.model_error = None
    except Exception as exc:  # noqa: BLE001 - expose startup failures through degraded health
        app.state.model_error = str(exc)
    yield


app = FastAPI(title="Fashion-MNIST Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    if inference_runtime.model is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": getattr(app.state, "model_error", "No model available yet."),
        }
    return {"status": "ok", "model_loaded": True, "model_key": inference_runtime.model_key}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:  # noqa: B008
    if file is None or file.filename is None:
        raise HTTPException(status_code=400, detail="An image file is required.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    payload = await file.read()
    if len(payload) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        prediction = inference_runtime.predict_image_bytes(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive validation
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "model_key": prediction["model_key"],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
