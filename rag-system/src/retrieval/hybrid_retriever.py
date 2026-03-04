"""
Hybrid retriever: combines vector search and BM25 keyword search via
Reciprocal Rank Fusion (RRF).
"""
from typing import Dict, List, Optional

from loguru import logger

from src.models import RetrievalResult
from src.retrieval.keyword_search import KeywordSearchEngine
from src.retrieval.vector_store import VectorStore

RRF_K = 60  # Standard constant for RRF scoring


class HybridRetriever:
    """
    Combines semantic vector search with BM25 keyword search using RRF.

    RRF formula (per document d):
        score(d) = Σ 1 / (k + rank_i(d))
    where k=60 and rank_i is the document's rank in list i.

    Args:
        vector_store: Pinecone or other vector DB interface.
        keyword_search: BM25 keyword search engine.
        alpha: Weight for vector search (0=keyword only, 1=vector only).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearchEngine,
        alpha: float = 0.5,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.alpha = alpha

    async def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,  # type: ignore[type-arg]
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval using RRF fusion.

        Algorithm:
        1. Get top_k*2 results from vector search.
        2. Get top_k*2 results from BM25 keyword search.
        3. Merge using RRF.
        4. Return top_k combined results.

        Args:
            query: Raw query string (for BM25).
            query_embedding: Dense query vector (for vector search).
            top_k: Final number of results to return.
            filters: Optional metadata filters for vector search.

        Returns:
            Merged and reranked list of RetrievalResult.
        """
        candidate_k = top_k * 2

        # -- Vector search --
        vector_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filters=filters,
        )

        # -- Keyword search (if index is built) --
        keyword_results: List[RetrievalResult] = []
        if self.keyword_search.is_built:
            kw_scored = self.keyword_search.search(query, top_k=candidate_k)
            keyword_results = self.keyword_search.to_retrieval_results(kw_scored)
        else:
            logger.debug("BM25 index not built; falling back to vector-only retrieval.")

        # -- RRF fusion --
        merged = self._rrf_merge(vector_results, keyword_results, top_k=top_k)

        logger.debug(
            "Hybrid retrieval complete",
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            merged=len(merged),
        )
        return merged

    def _rrf_merge(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Apply Reciprocal Rank Fusion to merge two result lists."""
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Score from vector results
        for rank, result in enumerate(vector_results, start=1):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.alpha / (RRF_K + rank)
            result_map[cid] = result

        # Score from keyword results
        for rank, result in enumerate(keyword_results, start=1):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - self.alpha) / (RRF_K + rank)
            if cid not in result_map:
                result_map[cid] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        merged: List[RetrievalResult] = []
        for cid in sorted_ids[:top_k]:
            result = result_map[cid]
            # Update score to RRF score (normalize to [0, 1] range)
            result = result.model_copy(update={"score": min(rrf_scores[cid] * RRF_K, 1.0)})
            merged.append(result)

        return merged
