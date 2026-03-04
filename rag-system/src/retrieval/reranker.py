"""
Cross-encoder reranker using sentence-transformers ms-marco model.
Improves retrieval precision by scoring query-document pairs directly.
"""
from typing import List

from loguru import logger

from src.models import RetrievalResult

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Reranks initial retrieval results using a cross-encoder model.

    Cross-encoders jointly encode query and document — they are slower than
    bi-encoders but produce significantly better relevance scores.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model_name = model_name
        self._model = CrossEncoder(model_name)
        logger.info("Reranker initialized", model=model_name)

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using the cross-encoder.

        Args:
            query: User query string.
            results: Initial retrieval results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Top-k results sorted by cross-encoder score descending,
            with updated `score` field normalized to [0, 1].
        """
        if not results:
            return []

        # Create (query, document) pairs for the cross-encoder
        pairs = [(query, result.content) for result in results]

        # Predict relevance scores (raw logits from ms-marco model)
        raw_scores = self._model.predict(pairs).tolist()  # type: ignore[union-attr,attr-defined]

        # Pair results with their raw scores
        scored = list(zip(results, raw_scores))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to [0, 1] using sigmoid-like scaling
        # ms-marco scores are typically in [-10, 10] range
        reranked: List[RetrievalResult] = []
        for result, raw_score in scored[:top_k]:
            # Sigmoid normalization
            import math
            normalized_score = 1.0 / (1.0 + math.exp(-raw_score / 5.0))
            reranked.append(result.model_copy(update={"score": round(normalized_score, 4)}))

        logger.debug(
            "Reranking complete",
            input_count=len(results),
            output_count=len(reranked),
        )
        return reranked
