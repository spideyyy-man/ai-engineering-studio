"""Domain exceptions + FastAPI exception handlers."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging import logger


class StudioError(Exception):
    status_code: int = 400

    def __init__(self, message: str, *, code: str = "studio_error", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class NotFoundError(StudioError):
    status_code = 404

    def __init__(self, what: str):
        super().__init__(f"{what} not found", code="not_found")


class ValidationError(StudioError):
    status_code = 422


class ProviderError(StudioError):
    """LLM / vector / external provider failure."""
    status_code = 502


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StudioError)
    async def _studio_handler(request: Request, exc: StudioError):
        logger.warning("studio_error {} on {}: {}", exc.code, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on {}: {}", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "message": str(exc)},
        )
