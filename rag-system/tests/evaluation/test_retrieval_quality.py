"""
Retrieval quality evaluation tests.
Measures Precision@K, Recall@K, MRR, and NDCG@K.

Run with: pytest tests/evaluation/test_retrieval_quality.py -v -m evaluation
"""
import json
import math
from pathlib import Path
from typing import List, Set

import pytest


GOLDEN_DATASET_PATH = Path("tests/evaluation/golden_dataset.json")


@pytest.fixture(scope="module")
def golden_dataset():
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Metric implementations
# ------------------------------------------------------------------

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Precision@K: fraction of top-K retrieved that are relevant."""
    retrieved_k = set(retrieved[:k])
    if k == 0:
        return 0.0
    return len(retrieved_k & relevant) / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Recall@K: fraction of relevant items retrieved in top K."""
    if not relevant:
        return 1.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """MRR: 1/rank of the first relevant result."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""

    def dcg(ids: List[str], rel: Set[str], k: int) -> float:
        return sum(
            (1 / math.log2(rank + 2)) if chunk_id in rel else 0
            for rank, chunk_id in enumerate(ids[:k])
        )

    actual_dcg = dcg(retrieved, relevant, k)
    # Ideal DCG: relevant items sorted first
    ideal = sorted(retrieved, key=lambda x: x in relevant, reverse=True)
    ideal_dcg = dcg(ideal, relevant, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ------------------------------------------------------------------
# Metric computation helpers
# ------------------------------------------------------------------

def compute_metrics(
    retrieved_ids: List[str],
    expected_chunks: List[str],
    k: int = 5,
) -> dict:
    """Compute all retrieval metrics for one query."""
    relevant = set(expected_chunks)
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant, k),
    }


# ------------------------------------------------------------------
# Unit tests for metric functions (no external deps needed)
# ------------------------------------------------------------------

class TestMetricFunctions:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], {"a", "b", "c"}, k=3) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)

    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_mrr_first_hit(self):
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_mrr_second_hit(self):
        assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_ndcg_perfect(self):
        score = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert score == pytest.approx(1.0)

    def test_ndcg_empty_relevant(self):
        score = ndcg_at_k(["a", "b"], set(), k=2)
        assert score == 0.0
