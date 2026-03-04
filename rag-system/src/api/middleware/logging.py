"""
Request logging middleware: logs all requests with UUID, latency, and status.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured per-request logging using loguru.

    Adds:
    - Unique request_id (UUID4)
    - HTTP method, path, status code
    - Latency in milliseconds
    - Client IP
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        request_id = str(uuid.uuid4())
        start = time.time()

        # Attach request_id to state so routes can access it
        request.state.request_id = request_id

        logger.bind(request_id=request_id).info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = round((time.time() - start) * 1000)
            logger.bind(request_id=request_id).error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise

        latency_ms = round((time.time() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency_ms)

        logger.bind(request_id=request_id).info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response
