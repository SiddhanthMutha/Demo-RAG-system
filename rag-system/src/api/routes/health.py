"""
Health check route: basic liveness and optional dependency checks.
"""
import time
from typing import Any, Dict

from fastapi import APIRouter

from src.models import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns service version and status. Does not check external dependencies
    (use /health/detailed for that).
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={"api": "healthy"},
    )


@router.get("/health/detailed", summary="Detailed health check")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Check health of all downstream dependencies.
    Returns individual status for each service.
    """
    from src.config import settings

    services: Dict[str, str] = {}
    overall = True

    # Check OpenAI reachability (lightweight by checking API key exists)
    services["openai_key_configured"] = (
        "ok" if settings.openai_api_key and not settings.openai_api_key.startswith("sk-placeholder") else "missing"
    )

    # Check Pinecone key configured
    services["pinecone_key_configured"] = (
        "ok" if settings.pinecone_api_key and settings.pinecone_api_key != "placeholder" else "missing"
    )

    if any(v != "ok" for v in services.values()):
        overall = False

    return {
        "status": "healthy" if overall else "degraded",
        "version": "1.0.0",
        "timestamp": time.time(),
        "services": services,
    }
