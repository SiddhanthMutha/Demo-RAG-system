"""
Evaluation API routes: run full-suite evaluations and inspect grouped history.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.database.models import ChunkRecord, EvaluationResult
from src.database.repository import EvaluationRepository
from src.evaluation.dataset import generate_synthetic_qa
from src.evaluation.harness import run_full_evaluation
from src.models import Chunk

router = APIRouter(prefix="/api/v1/eval", tags=["Evaluation"])

FULL_SUITE_METRICS = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg",
    "correctness_rate",
    "faithfulness_rate",
    "avg_latency_ms",
    "total_cost_usd",
    "num_queries",
]


class EvalRunRequest(BaseModel):
    queries: Optional[List[Dict[str, Any]]] = None
    synthetic: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    max_qa_pairs: int = Field(default=20, ge=1, le=100)
    sample_size: int = Field(default=20, ge=1, le=200)
    model: str = Field(default="gpt-3.5-turbo")


class EvalRunResponse(BaseModel):
    run_id: str
    suite_type: str = "full"
    dataset_type: str
    top_k: int
    model: str
    num_queries: int
    metrics: Dict[str, float]
    queries: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str


def _serialize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    created_at = run.get("created_at")
    return {
        "run_id": run["run_id"],
        "test_name": run.get("test_name", "full_eval"),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        "details": run.get("details", {}),
        "metrics": run.get("metrics", {}),
    }


async def _build_synthetic_queries(
    session: AsyncSession,
    max_qa_pairs: int,
    sample_size: int,
) -> List[Dict[str, Any]]:
    result = await session.execute(select(ChunkRecord).limit(sample_size))
    chunk_records = result.scalars().all()

    chunks = [
        Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            token_count=chunk.token_count,
            chunk_index=chunk.chunk_index,
            metadata=chunk.chunk_metadata or {},
        )
        for chunk in chunk_records
    ]
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks found for synthetic evaluation.")

    qa_pairs = await generate_synthetic_qa(
        chunks,
        max_qa_pairs=max_qa_pairs,
        sample_size=sample_size,
    )
    return [
        {
            "query": qa["question"],
            "expected_ids": [qa["chunk_id"]],
            "expected_answer": qa["answer"],
        }
        for qa in qa_pairs
        if qa.get("question") and qa.get("answer")
    ]


async def _persist_full_suite_run(
    session: AsyncSession,
    run_id: str,
    request: EvalRunRequest,
    metrics: Dict[str, float],
    query_count: int,
) -> None:
    repo = EvaluationRepository(session)
    base_details = {
        "run_id": run_id,
        "suite_type": "full",
        "dataset_type": "synthetic" if request.synthetic else "custom",
        "top_k": request.top_k,
        "num_queries": query_count,
        "model": request.model,
        "sample_size": request.sample_size,
        "max_qa_pairs": request.max_qa_pairs,
        "recorded_at": datetime.utcnow().isoformat(),
    }
    for metric_name in FULL_SUITE_METRICS:
        await repo.save_result(
            test_name="full_eval",
            metric_name=metric_name,
            metric_value=float(metrics.get(metric_name, 0.0)),
            details=base_details,
        )


async def execute_evaluation_run(
    request: EvalRunRequest,
    session: AsyncSession,
) -> EvalRunResponse:
    """Shared execution logic used by JSON and HTML endpoints."""
    queries = request.queries
    if request.synthetic and not queries:
        queries = await _build_synthetic_queries(
            session=session,
            max_qa_pairs=request.max_qa_pairs,
            sample_size=request.sample_size,
        )

    if not queries:
        raise HTTPException(status_code=422, detail="Provide queries or set synthetic=True.")

    run_id = str(uuid4())
    try:
        results = await run_full_evaluation(
            queries=queries,
            top_k=request.top_k,
            model=request.model,
        )
    except Exception as exc:
        logger.error("Evaluation run failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    metrics = results["metrics"]
    query_count = int(metrics.get("num_queries", len(results.get("queries", []))))

    try:
        await _persist_full_suite_run(
            session=session,
            run_id=run_id,
            request=request,
            metrics=metrics,
            query_count=query_count,
        )
    except Exception as exc:
        logger.warning("Failed to persist eval results", error=str(exc))

    return EvalRunResponse(
        run_id=run_id,
        dataset_type="synthetic" if request.synthetic else "custom",
        top_k=request.top_k,
        model=request.model,
        num_queries=query_count,
        metrics={name: float(metrics.get(name, 0.0)) for name in FULL_SUITE_METRICS},
        queries=results.get("queries", []),
        created_at=datetime.utcnow().isoformat(),
    )


@router.post("/run", response_model=EvalRunResponse, summary="Run full evaluation suite")
async def run_eval(
    request: EvalRunRequest,
    session: AsyncSession = Depends(get_session),
) -> EvalRunResponse:
    """Trigger a full evaluation run."""
    return await execute_evaluation_run(request, session)


@router.get("/results", summary="List raw evaluation metric rows")
async def list_results(
    session: AsyncSession = Depends(get_session),
) -> List[Dict[str, Any]]:
    """Return raw evaluation rows, most recent first."""
    repo = EvaluationRepository(session)
    rows = await repo.list_results(limit=100)
    return [
        {
            "id": row.id,
            "test_name": row.test_name,
            "metric_name": row.metric_name,
            "metric_value": row.metric_value,
            "details": row.details,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


@router.get("/runs", summary="List grouped evaluation runs")
async def list_grouped_runs(
    session: AsyncSession = Depends(get_session),
) -> List[Dict[str, Any]]:
    """Return grouped evaluation history keyed by run_id."""
    repo = EvaluationRepository(session)
    runs = await repo.list_grouped_runs(limit=20)
    return [_serialize_run(run) for run in runs]


@router.get("/runs/{run_id}", summary="Get a grouped evaluation run")
async def run_details(
    run_id: str,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Return one grouped evaluation run."""
    repo = EvaluationRepository(session)
    run = await repo.get_run_details(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return _serialize_run(run)


@router.get("/metrics", summary="Aggregated evaluation metrics")
async def aggregated_metrics(
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Return average metric values grouped by metric_name."""
    stmt = (
        select(
            EvaluationResult.metric_name,
            func.avg(EvaluationResult.metric_value).label("avg_value"),
            func.count(EvaluationResult.id).label("count"),
        )
        .group_by(EvaluationResult.metric_name)
    )
    result = await session.execute(stmt)
    rows = result.all()
    return {
        row.metric_name: {"avg": round(row.avg_value, 4), "count": row.count}
        for row in rows
    }
