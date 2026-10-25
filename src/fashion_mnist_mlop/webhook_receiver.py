from __future__ import annotations

from fastapi import FastAPI, Request

app = FastAPI(title="Webhook receiver")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def receive_webhook(request: Request) -> dict[str, object]:
    payload = await request.json()
    print(f"Webhook received: {payload}")
    return {"status": "ok", "received": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
