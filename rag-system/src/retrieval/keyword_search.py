"""
BM25 keyword search engine using rank-bm25.
Provides exact keyword matching as the keyword component of hybrid retrieval.
"""
from typing import List, Optional, Tuple

from loguru import logger

from src.models import Chunk, RetrievalResult


class KeywordSearchEngine:
    """
    In-memory BM25 index for keyword search.

    Must be rebuilt whenever chunks are added/removed.
    For production scale, consider an Elasticsearch backend.
    """

    def __init__(self) -> None:
        self._bm25: Optional[object] = None  # rank_bm25.BM25Okapi
        self._chunks: List[Chunk] = []

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Build or rebuild BM25 index from chunks.

        Args:
            chunks: All Chunk objects to index.
        """
        from rank_bm25 import BM25Okapi  # type: ignore

        if not chunks:
            logger.warning("No chunks provided to index_chunks; BM25 index not built.")
            return

        tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunks = chunks
        logger.info("BM25 index built", num_chunks=len(chunks))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Run BM25 keyword search.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (Chunk, normalized_score) tuples, sorted descending.
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index not built; returning empty results.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)  # type: ignore

        # Normalize scores to [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized = scores / max_score

        # Pair with chunks and sort
        results = sorted(
            zip(self._chunks, normalized.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return results[:top_k]

    def to_retrieval_results(
        self, scored_chunks: List[Tuple[Chunk, float]]
    ) -> List[RetrievalResult]:
        """Convert BM25 results to RetrievalResult objects."""
        return [
            RetrievalResult(
                chunk_id=chunk.id,
                content=chunk.content,
                score=float(score),
                metadata=chunk.metadata,
                document_source=chunk.metadata.get("source", ""),
            )
            for chunk, score in scored_chunks
        ]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    @property
    def is_built(self) -> bool:
        """True if the BM25 index has been built."""
        return self._bm25 is not None and len(self._chunks) > 0
