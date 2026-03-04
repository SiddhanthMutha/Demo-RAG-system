"""
Error handling middleware: consistent JSON error format for all exceptions.
"""
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Catch unhandled exceptions and return a consistent JSON error response.
    Masks internal details from the client while logging full context.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")

            logger.bind(request_id=request_id).error(
                "Unhandled exception",
                exc_type=type(exc).__name__,
                path=request.url.path,
                traceback=traceback.format_exc(),
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "detail": str(exc) if __debug__ else "An unexpected error occurred.",
                },
            )
