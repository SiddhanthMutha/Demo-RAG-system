"""
Unit tests for the evaluation harness.
"""
from unittest.mock import AsyncMock, patch

import pytest

from src.evaluation.harness import run_full_evaluation


class TestRunFullEvaluation:
    @pytest.mark.asyncio
    async def test_returns_zero_metrics_for_empty_queries(self):
        result = await run_full_evaluation([])
        assert result["suite_type"] == "full"
        assert result["metrics"]["num_queries"] == 0.0
        assert result["queries"] == []

    @pytest.mark.asyncio
    async def test_aggregates_retrieval_and_generation_metrics(self):
        fake_cases = [
            {
                "query": "q1",
                "retrieved_chunks": [],
                "answer": "a1",
                "latency_ms": 100,
                "tokens_used": 20,
                "cost_usd": 0.01,
                "model": "gpt-3.5-turbo",
            },
            {
                "query": "q2",
                "retrieved_chunks": [],
                "answer": "a2",
                "latency_ms": 200,
                "tokens_used": 30,
                "cost_usd": 0.02,
                "model": "gpt-3.5-turbo",
            },
        ]
        with patch(
            "src.evaluation.harness.run_single_case",
            new=AsyncMock(side_effect=fake_cases),
        ), patch(
            "src.evaluation.harness.precision_at_k",
            side_effect=[0.5, 1.0],
        ), patch(
            "src.evaluation.harness.recall_at_k",
            side_effect=[0.25, 1.0],
        ), patch(
            "src.evaluation.harness.mrr",
            side_effect=[0.5, 1.0],
        ), patch(
            "src.evaluation.harness.ndcg",
            side_effect=[0.4, 0.8],
        ), patch(
            "src.evaluation.harness.check_answer_correctness",
            side_effect=[True, False],
        ), patch(
            "src.evaluation.harness.check_faithfulness",
            new=AsyncMock(side_effect=[True, True]),
        ):
            result = await run_full_evaluation(
                [
                    {"query": "q1", "expected_ids": ["1"], "expected_answer": "a1"},
                    {"query": "q2", "expected_ids": ["2"], "expected_answer": "a2"},
                ]
            )

        assert result["metrics"]["precision_at_k"] == pytest.approx(0.75)
        assert result["metrics"]["recall_at_k"] == pytest.approx(0.625)
        assert result["metrics"]["mrr"] == pytest.approx(0.75)
        assert result["metrics"]["ndcg"] == pytest.approx(0.6)
        assert result["metrics"]["correctness_rate"] == pytest.approx(0.5)
        assert result["metrics"]["faithfulness_rate"] == pytest.approx(1.0)
        assert result["metrics"]["avg_latency_ms"] == pytest.approx(150.0)
        assert result["metrics"]["total_cost_usd"] == pytest.approx(0.03)
        assert result["metrics"]["num_queries"] == pytest.approx(2.0)
