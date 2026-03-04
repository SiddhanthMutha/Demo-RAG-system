"""
Database repository: async CRUD operations for all entities.
"""
import hashlib
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ChunkRecord, DocumentRecord, EvaluationResult, QueryLog
from src.models import Chunk, Document


class DocumentRepository:
    """CRUD operations on DocumentRecord and ChunkRecord."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _hash_content(content: str) -> str:
        """SHA-256 hash of document content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()

    async def document_exists(self, content: str) -> Optional[str]:
        """Return existing document_id if content already ingested, else None."""
        content_hash = self._hash_content(content)
        result = await self.session.execute(
            select(DocumentRecord.id).where(DocumentRecord.content_hash == content_hash)
        )
        row = result.scalar_one_or_none()
        return str(row) if row else None

    async def save_document(self, doc: Document) -> DocumentRecord:
        """Persist a Document to the database."""
        record = DocumentRecord(
            id=doc.id,
            content_hash=self._hash_content(doc.content),
            doc_type=doc.doc_type.value,
            source=doc.source,
            metadata=doc.metadata,
            created_at=doc.created_at,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def save_chunks(self, chunks: List[Chunk]) -> int:
        """Persist a list of Chunk objects. Returns count saved."""
        records = [
            ChunkRecord(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                embedding=chunk.embedding,
                token_count=chunk.token_count,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
        self.session.add_all(records)
        await self.session.flush()
        return len(records)

    async def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Fetch a DocumentRecord by ID."""
        result = await self.session.execute(
            select(DocumentRecord).where(DocumentRecord.id == document_id)
        )
        return result.scalar_one_or_none()

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks (cascade). Returns True if deleted."""
        record = await self.get_document(document_id)
        if record is None:
            return False
        await self.session.delete(record)
        await self.session.flush()
        return True


class QueryRepository:
    """CRUD operations on QueryLog."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log_query(
        self,
        query_id: str,
        query_text: str,
        response_text: str,
        retrieval_results: List[Dict[str, Any]],
        latency_ms: float,
        tokens_used: int,
        cost_usd: float,
        model_used: str,
    ) -> QueryLog:
        """Persist a query log entry."""
        log = QueryLog(
            id=query_id,
            query_text=query_text,
            response_text=response_text,
            retrieval_results=retrieval_results,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            model_used=model_used,
        )
        self.session.add(log)
        await self.session.flush()
        return log


class EvaluationRepository:
    """CRUD operations on EvaluationResult."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_result(
        self,
        test_name: str,
        metric_name: str,
        metric_value: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Persist an evaluation metric result."""
        result = EvaluationResult(
            id=str(uuid4()),
            test_name=test_name,
            metric_name=metric_name,
            metric_value=metric_value,
            details=details or {},
        )
        self.session.add(result)
        await self.session.flush()
        return result
