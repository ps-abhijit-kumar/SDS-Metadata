"""Global exception handlers for the FastAPI application.

Maps domain exceptions to appropriate HTTP status codes so the API always
returns structured JSON error responses rather than unhandled 500 stack traces.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain.exceptions.base import (
    ApplicationException,
    DocumentNotFoundException,
    InvalidDocumentException,
)

logger = logging.getLogger(__name__)


async def application_exception_handler(
    request: Request, exc: ApplicationException
) -> JSONResponse:
    """Handle all known application exceptions with appropriate HTTP codes."""
    if isinstance(exc, DocumentNotFoundException):
        status_code = 404
    elif isinstance(exc, InvalidDocumentException):
        status_code = 422
    else:
        status_code = 500
        logger.error(
            "ApplicationException | path=%s | type=%s | message=%s",
            request.url.path,
            type(exc).__name__,
            exc.message,
        )

    return JSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "detail": exc.message},
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler for unexpected exceptions."""
    logger.exception(
        "Unhandled exception | path=%s | type=%s",
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please check the server logs.",
        },
    )
