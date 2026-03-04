"""
SQLAlchemy ORM models for persistent storage.
Uses pgvector extension for embedding storage in PostgreSQL.
"""
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class DocumentRecord(Base):
    """Persisted document metadata."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    content_hash = Column(String, unique=True, index=True, nullable=False)
    doc_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("ChunkRecord", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<DocumentRecord id={self.id} type={self.doc_type}>"


class ChunkRecord(Base):
    """Persisted text chunk with optional pgvector embedding."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # Dimension matches all-MiniLM-L6-v2
    token_count = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    metadata = Column(JSON, default=dict)

    document = relationship("DocumentRecord", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ChunkRecord id={self.id} doc={self.document_id} idx={self.chunk_index}>"


class QueryLog(Base):
    """Log of all queries for monitoring and evaluation."""

    __tablename__ = "query_logs"

    id = Column(String, primary_key=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    retrieval_results = Column(JSON)  # [{chunk_id, score}, ...]
    latency_ms = Column(Float)
    tokens_used = Column(Integer)
    cost_usd = Column(Float)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} latency={self.latency_ms}ms>"


class EvaluationResult(Base):
    """Stored evaluation metric results."""

    __tablename__ = "evaluation_results"

    id = Column(String, primary_key=True)
    test_name = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<EvaluationResult {self.metric_name}={self.metric_value}>"
