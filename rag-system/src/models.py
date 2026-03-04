"""
Core Pydantic data models for the RAG system.
All API request/response types and domain entities are defined here.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types for ingestion."""

    PDF = "pdf"
    WEB = "web"
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SYNTAX_AWARE = "syntax_aware"  # For code files


class Document(BaseModel):
    """Raw document before processing."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., min_length=1)
    doc_type: DocumentType
    source: str = Field(..., description="File path, URL, or identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "Machine learning is a subset of AI...",
                "doc_type": "pdf",
                "source": "/data/ml_paper.pdf",
                "metadata": {"author": "John Doe", "pages": 12},
            }
        }
    }


class Chunk(BaseModel):
    """Processed chunk ready for embedding and indexing."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding has a supported dimension."""
        if v is not None and len(v) not in [384, 768, 1536]:
            raise ValueError(f"Embedding dimension {len(v)} must be 384, 768, or 1536")
        return v


class QueryRequest(BaseModel):
    """Incoming user query request."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    use_reranking: bool = Field(default=True)
    stream: bool = Field(default=True)
    model: Literal["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet"] = "gpt-3.5-turbo"

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "use_reranking": True,
                "stream": False,
                "model": "gpt-3.5-turbo",
            }
        }
    }


class RetrievalResult(BaseModel):
    """A single retrieved chunk with relevance score."""

    chunk_id: str
    content: str
    score: float = Field(..., ge=0, le=1)
    metadata: Dict[str, Any]
    document_source: str


class QueryResponse(BaseModel):
    """Complete non-streaming query response."""

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    sources: List[RetrievalResult]
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "tokens_used": 0,
            "latency_ms": 0,
            "cost_usd": 0.0,
            "model": "",
        }
    )


class IngestionRequest(BaseModel):
    """Document ingestion request body."""

    source: str = Field(..., description="File path or URL")
    doc_type: DocumentType
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    metadata: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "/data/report.pdf",
                "doc_type": "pdf",
                "chunking_strategy": "semantic",
            }
        }
    }


class IngestionResponse(BaseModel):
    """Result of a document ingestion request."""

    document_id: str
    chunks_created: int
    status: Literal["success", "partial", "failed"]
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: Dict[str, str] = Field(default_factory=dict)
