"""
Unit tests for evaluation metrics.
"""
import pytest

from src.evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg
from src.models import RetrievalResult


def _make_result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        score=score,
        metadata={},
        document_source=f"doc_{chunk_id}",
    )


class TestPrecisionAtK:
    def test_all_relevant(self):
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        assert precision_at_k(results, ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        assert precision_at_k(results, ["x", "y"], k=2) == 0.0

    def test_partial_relevant(self):
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        assert precision_at_k(results, ["a", "c"], k=3) == pytest.approx(2 / 3)

    def test_empty_results(self):
        assert precision_at_k([], ["a"], k=5) == 0.0

    def test_k_zero(self):
        results = [_make_result("a")]
        assert precision_at_k(results, ["a"], k=0) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        results = [_make_result("a"), _make_result("b")]
        assert recall_at_k(results, ["a", "b"], k=2) == 1.0

    def test_partial_found(self):
        results = [_make_result("a"), _make_result("b")]
        assert recall_at_k(results, ["a", "b", "c"], k=2) == pytest.approx(2 / 3)

    def test_none_found(self):
        results = [_make_result("x"), _make_result("y")]
        assert recall_at_k(results, ["a", "b"], k=2) == 0.0

    def test_empty_expected(self):
        results = [_make_result("a")]
        assert recall_at_k(results, [], k=2) == 0.0


class TestMRR:
    def test_first_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        assert mrr(results, ["a"]) == 1.0

    def test_second_relevant(self):
        results = [_make_result("x"), _make_result("a")]
        assert mrr(results, ["a"]) == 0.5

    def test_none_relevant(self):
        results = [_make_result("x"), _make_result("y")]
        assert mrr(results, ["a"]) == 0.0

    def test_empty_results(self):
        assert mrr([], ["a"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        assert ndcg(results, ["a", "b", "c"]) == pytest.approx(1.0)

    def test_no_relevant(self):
        results = [_make_result("x"), _make_result("y")]
        assert ndcg(results, ["a", "b"]) == 0.0

    def test_empty_results(self):
        assert ndcg([], ["a"]) == 0.0

    def test_partial_relevance(self):
        results = [_make_result("x"), _make_result("a"), _make_result("y"), _make_result("b")]
        score = ndcg(results, ["a", "b"])
        assert 0.0 < score < 1.0
