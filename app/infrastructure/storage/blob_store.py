"""Where the original bytes of an uploaded document are kept.

Search never reads these back (it uses the extracted text in the
database) -- they are retained for provenance/audit, so the only operation
the app needs is ``save``. ``LocalBlobStore`` keeps the previous behavior
(files under a directory, which on a hosted deployment must be a
persistent volume); ``S3BlobStore`` writes to any S3-compatible bucket so
the API containers stay stateless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def save(self, name: str, content: bytes) -> None:
        ...


class LocalBlobStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def save(self, name: str, content: bytes) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        (self._directory / name).write_bytes(content)


class S3BlobStore:
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None, client=None) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "upload_backend='s3' requires the optional 'boto3' package (see requirements-cloud.txt)"
                ) from exc
            client = boto3.client("s3", endpoint_url=endpoint_url)
        self._client = client
        self._bucket = bucket
        self._prefix = prefix

    def save(self, name: str, content: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=f"{self._prefix}{name}", Body=content)
