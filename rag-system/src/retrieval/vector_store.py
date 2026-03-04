"""
Pinecone vector store interface.
Provides upsert, semantic search, delete, and stats operations.
"""
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import Chunk, RetrievalResult


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""

    pass


class VectorStore:
    """
    Pinecone vector database interface.

    Wraps the Pinecone SDK to provide a clean async-friendly API.
    Note: Pinecone SDK calls are synchronous; we run them in thread pool.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> None:
        from pinecone import Pinecone  # type: ignore

        self._api_key = api_key or settings.pinecone_api_key
        self._index_name = index_name or settings.pinecone_index_name

        self._pc = Pinecone(api_key=self._api_key)
        self._index = self._pc.Index(self._index_name)
        logger.info("VectorStore connected", index=self._index_name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """
        Insert or update chunks in Pinecone.

        Args:
            chunks: Chunks that must already have `embedding` populated.

        Returns:
            Number of successfully upserted chunks.

        Raises:
            VectorStoreError: If chunks lack embeddings or upsert fails.
        """
        import asyncio

        missing = [c.id for c in chunks if c.embedding is None]
        if missing:
            raise VectorStoreError(f"Chunks missing embeddings: {missing}")

        vectors = [
            {
                "id": chunk.id,
                "values": chunk.embedding,
                "metadata": {
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "content": chunk.content[:1000],  # Store truncated for retrieval
                },
            }
            for chunk in chunks
        ]

        # Pinecone recommends batches of ≤100
        batch_size = 100
        loop = asyncio.get_event_loop()
        total = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            await loop.run_in_executor(None, lambda b=batch: self._index.upsert(vectors=b))
            total += len(batch)

        logger.info("Upserted chunks to Pinecone", count=total)
        return total

    async def delete_document(self, document_id: str) -> bool:
        """Delete all vectors belonging to a document via metadata filter."""
        import asyncio

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._index.delete(filter={"document_id": {"$eq": document_id}}),
            )
            logger.info("Deleted document vectors", document_id=document_id)
            return True
        except Exception as e:
            logger.error("Failed to delete document vectors", document_id=document_id, error=str(e))
            return False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Semantic similarity search.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filters: Pinecone metadata filter dict (e.g., {"doc_type": {"$eq": "pdf"}}).

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        query_kwargs: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filters:
            query_kwargs["filter"] = filters

        response = await loop.run_in_executor(
            None, lambda: self._index.query(**query_kwargs)
        )

        results: List[RetrievalResult] = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(
                RetrievalResult(
                    chunk_id=match.id,
                    content=meta.get("content", ""),
                    score=float(match.score),
                    metadata=meta,
                    document_source=meta.get("source", ""),
                )
            )

        return results

    async def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        import asyncio

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self._index.describe_index_stats)
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
        }
