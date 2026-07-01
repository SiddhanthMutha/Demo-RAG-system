"""
Evaluation metrics for RAG system.
Implements standard IR and RAG evaluation metrics.
"""
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.models import RetrievalResult

def _get_relevants(
    results: List["RetrievalResult"], expected_ids: List[str]
) -> List["RetrievalResult"]:
    """Filter results whose chunk_id is in expected_ids."""
    expected_set = set(expected_ids)
    return [r for r in results if r.chunk_id in expected_set]

def precision_at_k(
    results: List["RetrievalResult"], expected_ids: List[str], k: int
) -> float:
    """
    Calculate Precision@k.

    Returns:
        Proportion of top-k results that are relevant.
    """
    if not results or k <= 0:
        return 0.0
    top_k = results[:k]
    relevant = _get_relevants(top_k, expected_ids)
    return len(relevant) / k

def recall_at_k(results: List["RetrievalResult"], expected_ids: List[str], k: int) -> float:
    """
    Calculate Recall@k.

    Returns:
        Proportion of expected documents found in top-k results.
    """
    if not expected_ids:
        return 0.0
    top_k = results[:k]
    relevant = _get_relevants(top_k, expected_ids)
    return len(relevant) / len(expected_ids)

def mrr(results: List["RetrievalResult"], expected_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Returns:
        1 / rank_of_first_relevant if any, else 0.0.
    """
    expected_set = set(expected_ids)
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in expected_set:
            return 1.0 / rank
    return 0.0

def ndcg(results: List["RetrievalResult"], expected_ids: List[str]) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG).

    Assumes binary relevance (relevant if chunk_id in expected_ids).

    Returns:
        nDCG score in [0, 1].
    """
    from math import log2

    if not results or not expected_ids:
        return 0.0

    expected_set = set(expected_ids)

    # DCG
    dcg = 0.0
    for i, result in enumerate(results, start=1):
        if result.chunk_id in expected_set:
            dcg += 1.0 / log2(i + 1)

    # Ideal DCG
    ideal = min(len(expected_ids), len(results))
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal + 1))

    return dcg / idcg if idcg > 0 else 0.0
