from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

load_dotenv()


class MinioArtifactStore:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool = False,
        bucket_name: str | None = None,
        model_prefix: str | None = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = secure or str(os.getenv("MINIO_SECURE", "false")).lower() == "true"
        self.bucket_name = bucket_name or os.getenv("MODEL_BUCKET_NAME", "fashion-minst-bucket")
        self.model_prefix = (model_prefix or os.getenv("MODEL_PREFIX", "model/")).strip("/")
        self.client = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=self.secure)

    def ensure_bucket(self) -> None:
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)

    def upload_bytes(self, object_key: str, payload: bytes, content_type: str = "application/octet-stream") -> str:
        self.ensure_bucket()
        self.client.put_object(
            self.bucket_name,
            object_key,
            io.BytesIO(payload),
            length=len(payload),
            content_type=content_type,
        )
        return object_key

    def upload_file(self, file_path: str | Path, object_key: str) -> str:
        path = Path(file_path)
        return self.upload_bytes(object_key, path.read_bytes())

    def download_bytes(self, object_key: str) -> bytes:
        response = self.client.get_object(self.bucket_name, object_key)
        with response:
            return response.read()

    def list_objects(self, prefix: str | None = None) -> list[str]:
        resolved_prefix = prefix or self.model_prefix
        objects = self.client.list_objects(self.bucket_name, prefix=f"{resolved_prefix}/" if not resolved_prefix.endswith("/") else resolved_prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def latest_object_key(self, prefix: str | None = None) -> str | None:
        resolved_prefix = prefix or self.model_prefix
        objects = self.client.list_objects(self.bucket_name, prefix=f"{resolved_prefix}/" if not resolved_prefix.endswith("/") else resolved_prefix, recursive=True)
        sorted_objects = sorted(
            objects,
            key=lambda obj: obj.last_modified if getattr(obj, "last_modified", None) is not None else datetime.min,
            reverse=True,
        )
        if not sorted_objects:
            return None
        return sorted_objects[0].object_name

    @staticmethod
    def from_env() -> MinioArtifactStore:
        return MinioArtifactStore(
            endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=str(os.getenv("MINIO_SECURE", "false")).lower() == "true",
            bucket_name=os.getenv("MODEL_BUCKET_NAME", "fashion-minst-bucket"),
            model_prefix=os.getenv("MODEL_PREFIX", "model/"),
        )


def ensure_bucket_exists(store: MinioArtifactStore) -> None:
    try:
        store.ensure_bucket()
    except S3Error as exc:  # pragma: no cover - only for runtime safety
        raise RuntimeError(f"Unable to initialize MinIO bucket: {exc}") from exc
