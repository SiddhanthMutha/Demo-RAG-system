"""
Unit tests for evaluation result grouping.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.database.repository import EvaluationRepository


class TestEvaluationRepositoryGrouping:
    @pytest.mark.asyncio
    async def test_groups_rows_by_run_id(self):
        rows = [
            SimpleNamespace(
                id="1",
                test_name="full_eval",
                metric_name="precision_at_k",
                metric_value=0.5,
                details={"run_id": "run-1", "dataset_type": "synthetic"},
                created_at=datetime(2026, 7, 1, 12, 0),
            ),
            SimpleNamespace(
                id="2",
                test_name="full_eval",
                metric_name="recall_at_k",
                metric_value=0.7,
                details={"run_id": "run-1", "dataset_type": "synthetic"},
                created_at=datetime(2026, 7, 1, 12, 0),
            ),
            SimpleNamespace(
                id="3",
                test_name="full_eval",
                metric_name="precision_at_k",
                metric_value=0.2,
                details={"run_id": "run-2", "dataset_type": "synthetic"},
                created_at=datetime(2026, 7, 1, 11, 0),
            ),
        ]
        repo = EvaluationRepository(session=AsyncMock())
        with patch.object(repo, "list_results", new=AsyncMock(return_value=rows)):
            runs = await repo.list_grouped_runs(limit=10)

        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-1"
        assert runs[0]["metrics"]["precision_at_k"] == 0.5
        assert runs[0]["metrics"]["recall_at_k"] == 0.7
