"""
Integration tests for HTML views, partials, and evaluation routes.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.routes.eval import EvalRunResponse
from src.database import get_session
from src.database.repository import DocumentRepository, EvaluationRepository
from src.models import IngestionResponse


@pytest.fixture
def app_with_fake_session():
    from src.api.main import app

    async def fake_session():
        yield AsyncMock()

    app.dependency_overrides[get_session] = fake_session
    yield app
    app.dependency_overrides.clear()


class TestViewRoutes:
    @pytest.mark.asyncio
    async def test_query_page_renders(self, app_with_fake_session):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_fake_session),
            base_url="http://test",
        ) as client:
            response = await client.get("/web/query")
        assert response.status_code == 200
        assert "Stream Answer" in response.text

    @pytest.mark.asyncio
    async def test_ingest_page_renders(self, app_with_fake_session):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_fake_session),
            base_url="http://test",
        ) as client:
            response = await client.get("/web/ingest")
        assert response.status_code == 200
        assert "Ingestion" in response.text

    @pytest.mark.asyncio
    async def test_documents_page_renders(self, app_with_fake_session):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_fake_session),
            base_url="http://test",
        ) as client:
            response = await client.get("/web/documents")
        assert response.status_code == 200
        assert "Documents" in response.text

    @pytest.mark.asyncio
    async def test_eval_page_renders(self, app_with_fake_session):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_fake_session),
            base_url="http://test",
        ) as client:
            response = await client.get("/web/eval")
        assert response.status_code == 200
        assert "Evaluation Dashboard" in response.text

    @pytest.mark.asyncio
    async def test_query_partial_renders_stream_shell(self, app_with_fake_session):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_fake_session),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/web/query/run",
                data={"query": "What is RAG?", "top_k": "5", "model": "gpt-3.5-turbo"},
            )
        assert response.status_code == 200
        assert "data-query-stream" in response.text


class TestViewPartials:
    @pytest.mark.asyncio
    async def test_documents_table_partial_renders_rows(self, app_with_fake_session):
        fake_documents = [
            {
                "id": "doc-1",
                "source": "handbook.pdf",
                "doc_type": "pdf",
                "created_at": datetime(2026, 7, 1, 10, 30),
                "chunk_count": 4,
            }
        ]
        with patch.object(
            DocumentRepository,
            "list_documents",
            new=AsyncMock(return_value=fake_documents),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app_with_fake_session),
                base_url="http://test",
            ) as client:
                response = await client.get("/web/documents/table")
        assert response.status_code == 200
        assert "handbook.pdf" in response.text
        assert "doc-1" in response.text

    @pytest.mark.asyncio
    async def test_ingest_source_partial_renders_result_card(self, app_with_fake_session):
        fake_result = IngestionResponse(
            document_id="doc-1",
            chunks_created=3,
            status="success",
            errors=[],
        )
        with patch("src.api.routes.views.perform_ingestion", new=AsyncMock(return_value=fake_result)):
            async with AsyncClient(
                transport=ASGITransport(app=app_with_fake_session),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/web/ingest/source",
                    data={
                        "source": "handbook.pdf",
                        "doc_type": "pdf",
                        "chunking_strategy": "semantic",
                    },
                )
        assert response.status_code == 200
        assert "doc-1" in response.text
        assert "3 chunks" in response.text

    @pytest.mark.asyncio
    async def test_eval_run_partial_updates_results_and_history(self, app_with_fake_session):
        fake_response = EvalRunResponse(
            run_id="run-1",
            dataset_type="synthetic",
            top_k=5,
            model="gpt-3.5-turbo",
            num_queries=2,
            metrics={
                "precision_at_k": 0.5,
                "recall_at_k": 0.6,
                "mrr": 0.7,
                "ndcg": 0.8,
                "correctness_rate": 1.0,
                "faithfulness_rate": 1.0,
                "avg_latency_ms": 120.0,
                "total_cost_usd": 0.02,
                "num_queries": 2.0,
            },
            queries=[
                {
                    "query": "What is RAG?",
                    "answer": "Retrieval augmented generation.",
                    "precision_at_k": 0.5,
                    "recall_at_k": 0.6,
                    "mrr": 0.7,
                    "ndcg": 0.8,
                    "correctness": True,
                    "faithfulness": True,
                }
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        fake_runs = [
            {
                "run_id": "run-1",
                "test_name": "full_eval",
                "created_at": datetime(2026, 7, 1, 12, 0),
                "details": {"dataset_type": "synthetic", "model": "gpt-3.5-turbo"},
                "metrics": {"precision_at_k": 0.5, "correctness_rate": 1.0},
            }
        ]
        with patch(
            "src.api.routes.views.execute_evaluation_run",
            new=AsyncMock(return_value=fake_response),
        ), patch.object(
            EvaluationRepository,
            "list_grouped_runs",
            new=AsyncMock(return_value=fake_runs),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app_with_fake_session),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/web/eval/run",
                    data={
                        "synthetic": "true",
                        "top_k": "5",
                        "model": "gpt-3.5-turbo",
                        "max_qa_pairs": "20",
                        "sample_size": "20",
                    },
                )
        assert response.status_code == 200
        assert "run-1" in response.text
        assert "hx-swap-oob" in response.text


class TestSSEEndpoint:
    @pytest.mark.asyncio
    async def test_sse_endpoint_exists(self, app_with_fake_session):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_fake_session),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/v1/query/stream/sse", params={"query": "test"})
        assert response.status_code != 404


class TestEvalAPIRoutes:
    @pytest.mark.asyncio
    async def test_eval_results_endpoint_exists(self, app_with_fake_session):
        with patch.object(EvaluationRepository, "list_results", new=AsyncMock(return_value=[])):
            async with AsyncClient(
                transport=ASGITransport(app=app_with_fake_session),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/eval/results")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_eval_runs_endpoint_returns_grouped_history(self, app_with_fake_session):
        fake_runs = [
            {
                "run_id": "run-1",
                "test_name": "full_eval",
                "created_at": datetime(2026, 7, 1, 12, 0),
                "details": {"dataset_type": "synthetic"},
                "metrics": {"precision_at_k": 0.5},
            }
        ]
        with patch.object(
            EvaluationRepository,
            "list_grouped_runs",
            new=AsyncMock(return_value=fake_runs),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app_with_fake_session),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/eval/runs")
        assert response.status_code == 200
        assert response.json()[0]["run_id"] == "run-1"

    @pytest.mark.asyncio
    async def test_documents_api_exists(self, app_with_fake_session):
        with patch.object(DocumentRepository, "list_documents", new=AsyncMock(return_value=[])):
            async with AsyncClient(
                transport=ASGITransport(app=app_with_fake_session),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/documents")
        assert response.status_code == 200
