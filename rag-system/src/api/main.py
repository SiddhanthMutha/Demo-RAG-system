"""
FastAPI application entry point.
Configures middleware, routers, lifespan events, and API metadata.
"""
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.middleware.error_handling import ErrorHandlingMiddleware
from src.api.middleware.logging import LoggingMiddleware
from src.api.routes import health, ingest, query
from src.config import settings


# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------

def _configure_logging() -> None:
    """Configure loguru with structured output."""
    logger.remove()
    if settings.log_format == "json":
        logger.add(
            sys.stdout,
            format=(
                '{"time": "{time:YYYY-MM-DDTHH:mm:ss.SSSZ}", '
                '"level": "{level}", "message": "{message}", {extra}}'
            ),
            level=settings.log_level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level=settings.log_level,
        )


# ------------------------------------------------------------------
# Lifespan event handler
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown events."""
    _configure_logging()
    logger.info("RAG System API starting up", version="1.0.0")

    # Initialize database (creates tables + pgvector extension)
    try:
        from src.database import init_db
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database initialization failed (may not be available)", error=str(e))

    yield  # Application is running

    logger.info("RAG System API shutting down")


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="Multi-Source RAG System",
    description=(
        "Production-quality Retrieval-Augmented Generation system "
        "with multi-format document ingestion, hybrid search, streaming responses, "
        "and automated evaluation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Middleware (order matters: outermost first in FastAPI)
# ------------------------------------------------------------------
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


# ------------------------------------------------------------------
# Root redirect
# ------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> dict:  # type: ignore[type-arg]
    return {
        "message": "Multi-Source RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
