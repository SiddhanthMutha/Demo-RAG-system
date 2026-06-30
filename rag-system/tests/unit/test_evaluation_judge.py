"""
Unit tests for evaluation judge functions.
"""
import pytest

from src.evaluation.judge import check_answer_correctness, check_faithfulness
from src.models import RetrievalResult


class TestCheckAnswerCorrectness:
    def test_exact_match(self):
        assert check_answer_correctness("Machine learning is AI", "Machine learning is AI") is True

    def test_substring_match(self):
        assert check_answer_correctness("AI", "Machine learning is a subset of AI") is True

    def test_case_insensitive(self):
        assert check_answer_correctness("Machine Learning", "machine learning") is True

    def test_different_answer(self):
        assert check_answer_correctness("Quantum physics", "Baking a cake") is False

    def test_empty_expected(self):
        assert check_answer_correctness("", "some answer") is False

    def test_token_overlap(self):
        assert check_answer_correctness(
            "deep learning uses neural networks",
            "deep learning uses neural networks with many layers"
        ) is True


class TestCheckFaithfulnessHeuristic:
    @pytest.mark.asyncio
    async def test_faithful_answer(self):
        chunks = [
            RetrievalResult(
                chunk_id="1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.9,
                metadata={},
                document_source="doc1",
            )
        ]
        result = await check_faithfulness(chunks, "Machine learning is a subset of AI.", method="heuristic")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        result = await check_faithfulness([], "Some answer", method="heuristic")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_answer(self):
        chunks = [
            RetrievalResult(
                chunk_id="1",
                content="Some context",
                score=0.9,
                metadata={},
                document_source="doc1",
            )
        ]
        result = await check_faithfulness(chunks, "", method="heuristic")
        assert result is False
