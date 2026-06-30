"""
Database repository: async CRUD operations for all entities.
"""
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import func, select
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
            doc_metadata=doc.metadata,
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
                chunk_metadata=chunk.metadata,
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

    async def list_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return documents with chunk counts, newest first."""
        stmt = (
            select(
                DocumentRecord.id,
                DocumentRecord.source,
                DocumentRecord.doc_type,
                DocumentRecord.created_at,
                func.count(ChunkRecord.id).label("chunk_count"),
            )
            .outerjoin(ChunkRecord, ChunkRecord.document_id == DocumentRecord.id)
            .group_by(
                DocumentRecord.id,
                DocumentRecord.source,
                DocumentRecord.doc_type,
                DocumentRecord.created_at,
            )
            .order_by(DocumentRecord.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        rows = result.all()
        return [
            {
                "id": row.id,
                "source": row.source,
                "doc_type": row.doc_type,
                "created_at": row.created_at,
                "chunk_count": int(row.chunk_count or 0),
            }
            for row in rows
        ]


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

    async def list_results(self, limit: int = 100) -> List[EvaluationResult]:
        """Return raw evaluation rows, newest first."""
        result = await self.session.execute(
            select(EvaluationResult)
            .order_by(EvaluationResult.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_grouped_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Group flat metric rows into evaluation runs using details.run_id.

        Rows without run_id are grouped under their record id so legacy data remains visible.
        """
        rows = await self.list_results(limit=max(limit * 12, 100))
        grouped: Dict[str, List[EvaluationResult]] = defaultdict(list)
        for row in rows:
            run_id = (row.details or {}).get("run_id") or row.id
            grouped[str(run_id)].append(row)

        runs: List[Dict[str, Any]] = []
        for run_id, run_rows in grouped.items():
            ordered_rows = sorted(
                run_rows,
                key=lambda item: item.created_at.isoformat() if item.created_at else "",
                reverse=True,
            )
            first = ordered_rows[0]
            details = first.details or {}
            metrics = {row.metric_name: row.metric_value for row in ordered_rows}
            runs.append(
                {
                    "run_id": run_id,
                    "test_name": first.test_name,
                    "created_at": first.created_at,
                    "details": details,
                    "metrics": metrics,
                }
            )

        runs.sort(
            key=lambda item: item["created_at"].isoformat() if item["created_at"] else "",
            reverse=True,
        )
        return runs[:limit]

    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a single grouped run if present."""
        runs = await self.list_grouped_runs(limit=100)
        for run in runs:
            if run["run_id"] == run_id:
                return run
        return None
