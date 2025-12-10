"""Google Cloud Storage helpers for BlueprintRecipe API."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

DEFAULT_BUCKET = "blueprint-8c1ca.appspot.com"
DEFAULT_PREFIX = "scenes"


class StorageUploadError(Exception):
    """Raised when an upload to cloud storage fails."""


async def upload_file_to_gcs(
    file: UploadFile,
    scene_id: str,
    bucket_name: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    client: Optional["storage.Client"] = None,
) -> str:
    """Upload an :class:`UploadFile` to Cloud Storage and return the gs:// URI.

    Parameters
    ----------
    file:
        The incoming FastAPI ``UploadFile`` to persist.
    scene_id:
        Scene identifier used to scope the upload path.
    bucket_name:
        Destination bucket name.
    prefix:
        Key prefix; defaults to ``scenes``.
    client:
        Optional storage client (useful for testing).

    Returns
    -------
    str
        The ``gs://`` URI of the uploaded object.

    Raises
    ------
    ValueError
        If ``scene_id`` or filename contain invalid path characters.
    StorageUploadError
        If the upload fails for any reason.
    RuntimeError
        If the Google Cloud Storage client is unavailable.
    """

    if not scene_id:
        raise ValueError("scene_id is required for upload")

    if "/" in scene_id or ".." in scene_id:
        raise ValueError("scene_id contains invalid path characters")

    filename = Path(file.filename or "upload.bin").name
    if filename in {"", ".", ".."}:
        raise ValueError("filename is required for upload")

    if "/" in filename or ".." in filename:
        raise ValueError("filename contains invalid path characters")

    if client is None:
        spec = importlib.util.find_spec("google.cloud.storage")
        if spec is None:
            raise RuntimeError("google-cloud-storage is not installed")

        storage_module = importlib.import_module("google.cloud.storage")
        client = storage_module.Client()
    bucket = client.bucket(bucket_name)

    object_path = f"{prefix}/{scene_id}/images/{filename}"
    blob = bucket.blob(object_path)

    try:
        # Ensure the underlying file handle is at the start before uploading.
        file.file.seek(0)
        await asyncio.to_thread(
            blob.upload_from_file,
            file.file,
            content_type=file.content_type,
        )
    except Exception as exc:  # pragma: no cover - network interaction
        raise StorageUploadError(f"Failed to upload to GCS: {exc}") from exc

    return f"gs://{bucket_name}/{object_path}"


__all__ = [
    "StorageUploadError",
    "upload_file_to_gcs",
]
