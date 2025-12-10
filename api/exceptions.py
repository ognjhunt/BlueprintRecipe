"""Custom exception types for BlueprintRecipe API."""

from __future__ import annotations

from fastapi import HTTPException


class BlueprintAPIError(Exception):
    """Base class for API-layer errors with HTTP metadata."""

    status_code: int = 500
    user_message: str = "An unexpected error occurred"

    def __init__(self, message: str | None = None, *, status_code: int | None = None):
        super().__init__(message or self.user_message)
        if status_code is not None:
            self.status_code = status_code
        if message is not None:
            self.user_message = message

    def to_http_exception(self) -> HTTPException:
        """Convert the error to an :class:`HTTPException`."""

        return HTTPException(status_code=self.status_code, detail=self.user_message)


class PlanningError(BlueprintAPIError):
    """Raised when scene planning fails."""

    status_code = 400
    user_message = "Scene planning failed"


class MatchingError(BlueprintAPIError):
    """Raised when asset matching fails."""

    status_code = 422
    user_message = "Asset matching failed"


class CompilationError(BlueprintAPIError):
    """Raised when recipe compilation fails."""

    status_code = 500
    user_message = "Recipe compilation failed"


class StorageError(BlueprintAPIError):
    """Raised when storage operations fail."""

    status_code = 502
    user_message = "Storage operation failed"


__all__ = [
    "BlueprintAPIError",
    "CompilationError",
    "MatchingError",
    "PlanningError",
    "StorageError",
]
