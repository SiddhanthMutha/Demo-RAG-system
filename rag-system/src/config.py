"""
Multi-Source RAG System
Configuration management via pydantic-settings.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # LLM APIs
    openai_api_key: str = "sk-placeholder"
    anthropic_api_key: Optional[str] = None

    # Vector DB (Pinecone)
    pinecone_api_key: str = "placeholder"
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "rag-system"

    # Database
    database_url: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # RAG
    default_top_k: int = 5
    max_context_tokens: int = 3000
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    # Feature flags
    use_reranking: bool = True
    use_hybrid_search: bool = True
    enable_query_logging: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
