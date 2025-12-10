"""Google Cloud Storage helpers for BlueprintRecipe API."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile

DEFAULT_BUCKET = "blueprint-8c1ca.appspot.com"
DEFAULT_PREFIX = "scenes"


class StorageUploadError(Exception):
    """Raised when an upload to cloud storage fails."""


def _get_storage_client(client: Optional["storage.Client"]):
    """Load a Google Cloud Storage client when not provided."""

    if client is not None:
        return client

    spec = importlib.util.find_spec("google.cloud.storage")
    if spec is None:
        raise RuntimeError("google-cloud-storage is not installed")

    storage_module = importlib.import_module("google.cloud.storage")
    return storage_module.Client()


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

    client = _get_storage_client(client)
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


async def upload_artifacts(
    job_id: str,
    artifacts: dict[str, Any],
    *,
    bucket_name: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    client: Optional["storage.Client"] = None,
) -> dict[str, Any]:
    """Upload compiled artifacts for a job to Cloud Storage.

    The helper supports nested artifact structures containing files, directories,
    lists, and dictionaries. Directories are uploaded recursively and represented
    as mappings of relative file paths to their corresponding GCS URIs.
    """

    if not job_id:
        raise ValueError("job_id is required for artifact upload")

    if "/" in job_id or ".." in job_id:
        raise ValueError("job_id contains invalid path characters")

    client = _get_storage_client(client)
    bucket = client.bucket(bucket_name)
    base_prefix = f"{prefix}/{job_id}/artifacts"

    async def _upload_local_file(local_path: Path, object_path: str) -> str:
        try:
            blob = bucket.blob(object_path)
            await asyncio.to_thread(blob.upload_from_filename, str(local_path))
        except Exception as exc:  # pragma: no cover - network interaction
            raise StorageUploadError(f"Failed to upload {local_path} to GCS: {exc}") from exc

        return f"gs://{bucket_name}/{object_path}"

    async def _process_value(value: Any, key_path: list[str]):
        if isinstance(value, dict):
            return {k: await _process_value(v, key_path + [k]) for k, v in value.items()}

        if isinstance(value, list):
            return [await _process_value(item, key_path + [str(idx)]) for idx, item in enumerate(value)]

        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")

        object_prefix = "/".join([base_prefix, *key_path]).rstrip("/")

        if path.is_dir():
            uploads: dict[str, str] = {}
            for file_path in path.rglob("*"):
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(path)
                object_path = f"{object_prefix}/{relative.as_posix()}"
                uploads[str(relative)] = await _upload_local_file(file_path, object_path)

            return {
                "base_uri": f"gs://{bucket_name}/{object_prefix}",
                "files": uploads,
            }

        object_path = f"{object_prefix}/{path.name}"
        return await _upload_local_file(path, object_path)

    uploaded: dict[str, Any] = {}
    for artifact_key, artifact_value in artifacts.items():
        uploaded[artifact_key] = await _process_value(artifact_value, [artifact_key])

    return uploaded


__all__ = [
    "StorageUploadError",
    "upload_artifacts",
    "upload_file_to_gcs",
]
