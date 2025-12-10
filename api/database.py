"""Database abstraction for job persistence."""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from typing import Any, Optional


class JobRepository(ABC):
    """Abstract repository for job persistence."""

    @abstractmethod
    async def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Persist a newly created job."""

    @abstractmethod
    async def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update a job and return the saved record."""

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a job by ID."""


class InMemoryJobRepository(JobRepository):
    """Simple in-memory repository suitable for development and testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            job_copy = copy.deepcopy(job)
            self._jobs[job_copy["job_id"]] = job_copy
            return copy.deepcopy(job_copy)

    async def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job '{job_id}' not found")

            current = copy.deepcopy(self._jobs[job_id])
            current.update(updates)
            self._jobs[job_id] = current
            return copy.deepcopy(current)

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        async with self._lock:
            if job_id not in self._jobs:
                return None
            return copy.deepcopy(self._jobs[job_id])


class FirestoreJobRepository(JobRepository):
    """Firestore-backed repository.

    The Firestore client is imported lazily to avoid adding dependencies when the
    backend is not selected. This implementation requires the
    ``google-cloud-firestore`` package to be installed and environment variables
    for credentials to be configured according to Google Cloud guidance.
    """

    def __init__(self, project_id: str, collection: str = "jobs") -> None:
        try:
            from google.cloud import firestore  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Firestore backend selected but google-cloud-firestore is not installed"
            ) from exc

        self._client = firestore.Client(project=project_id)
        self._collection = self._client.collection(collection)

    async def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        doc_ref = self._collection.document(job["job_id"])
        await loop.run_in_executor(None, doc_ref.set, job)
        return copy.deepcopy(job)

    async def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        doc_ref = self._collection.document(job_id)
        snapshot = await loop.run_in_executor(None, doc_ref.get)
        if not snapshot.exists:
            raise KeyError(f"Job '{job_id}' not found")

        await loop.run_in_executor(None, doc_ref.update, updates)
        merged = {**snapshot.to_dict(), **updates}
        return copy.deepcopy(merged)

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(
            None, self._collection.document(job_id).get
        )
        if not snapshot.exists:
            return None
        return copy.deepcopy(snapshot.to_dict())


__all__ = [
    "JobRepository",
    "InMemoryJobRepository",
    "FirestoreJobRepository",
]
