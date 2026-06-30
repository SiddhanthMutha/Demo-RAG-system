"""
Evaluation harness for full-suite RAG evaluation.
"""
import time
from typing import Any, Dict, List, Optional

from src.generation.context_manager import ContextManager
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from src.models import RetrievalResult
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

from .judge import check_answer_correctness, check_faithfulness
from .metrics import mrr, ndcg, precision_at_k, recall_at_k

_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStore] = None


def _get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _resolve_provider(model: str) -> tuple[str, str]:
    """Map UI model names to provider + concrete model values."""
    if model == "claude-3-5-sonnet":
        return "anthropic", "claude-3-5-sonnet-20241022"
    return "openai", model


async def _run_retrieval(query: str, top_k: int = 10) -> List[RetrievalResult]:
    """Embed query and perform vector search."""
    emb_service = _get_embedding_service()
    query_embedding = (await emb_service.embed_batch([query]))[0]
    vs = _get_vector_store()
    return await vs.search(query_embedding=query_embedding, top_k=top_k * 2)


async def run_single_case(
    query_text: str,
    top_k: int = 5,
    model: str = "gpt-3.5-turbo",
) -> Dict[str, Any]:
    """Run retrieval + generation for one evaluation case."""
    start = time.time()

    retrieved = await _run_retrieval(query_text, top_k=top_k)
    provider, resolved_model = _resolve_provider(model)

    ctx_manager = ContextManager(model_name=resolved_model)
    selected_chunks, _ = ctx_manager.fit_context(
        chunks=retrieved,
        system_prompt=SYSTEM_PROMPT,
        query=query_text,
    )
    if not selected_chunks:
        selected_chunks = retrieved[:top_k]

    prompt = build_rag_prompt(query=query_text, retrieved_chunks=selected_chunks)
    llm = LLMClient(provider=provider, model=resolved_model)
    answer, input_tokens, output_tokens = await llm.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    )

    latency_ms = round((time.time() - start) * 1000)
    return {
        "query": query_text,
        "retrieved_chunks": selected_chunks,
        "answer": answer,
        "latency_ms": latency_ms,
        "tokens_used": input_tokens + output_tokens,
        "cost_usd": llm.estimate_cost(input_tokens, output_tokens),
        "model": resolved_model,
    }


async def run_full_evaluation(
    queries: List[Dict[str, Any]],
    top_k: int = 5,
    model: str = "gpt-3.5-turbo",
    faithfulness_method: str = "heuristic",
) -> Dict[str, Any]:
    """Run a full evaluation suite across retrieval, generation, and operational metrics."""
    per_query_results: List[Dict[str, Any]] = []

    for query_record in queries:
        query_text = query_record["query"]
        expected_ids = query_record.get("expected_ids", [])
        expected_answer = query_record.get("expected_answer", "")

        result = await run_single_case(query_text, top_k=top_k, model=model)
        retrieved = result["retrieved_chunks"]
        answer = result["answer"]

        per_query_results.append(
            {
                "query": query_text,
                "answer": answer,
                "expected_ids": expected_ids,
                "precision_at_k": precision_at_k(retrieved, expected_ids, k=top_k),
                "recall_at_k": recall_at_k(retrieved, expected_ids, k=top_k),
                "mrr": mrr(retrieved, expected_ids),
                "ndcg": ndcg(retrieved, expected_ids),
                "correctness": check_answer_correctness(expected_answer, answer)
                if expected_answer
                else False,
                "faithfulness": await check_faithfulness(
                    retrieved,
                    answer,
                    method=faithfulness_method,
                ),
                "latency_ms": result["latency_ms"],
                "tokens_used": result["tokens_used"],
                "cost_usd": result["cost_usd"],
                "source_count": len(retrieved),
            }
        )

    if not per_query_results:
        return {
            "suite_type": "full",
            "metrics": {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "correctness_rate": 0.0,
                "faithfulness_rate": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost_usd": 0.0,
                "num_queries": 0.0,
            },
            "queries": [],
        }

    count = len(per_query_results)
    metrics = {
        "precision_at_k": sum(r["precision_at_k"] for r in per_query_results) / count,
        "recall_at_k": sum(r["recall_at_k"] for r in per_query_results) / count,
        "mrr": sum(r["mrr"] for r in per_query_results) / count,
        "ndcg": sum(r["ndcg"] for r in per_query_results) / count,
        "correctness_rate": sum(1 for r in per_query_results if r["correctness"]) / count,
        "faithfulness_rate": sum(1 for r in per_query_results if r["faithfulness"]) / count,
        "avg_latency_ms": sum(r["latency_ms"] for r in per_query_results) / count,
        "total_cost_usd": sum(r["cost_usd"] for r in per_query_results),
        "num_queries": float(count),
    }

    return {
        "suite_type": "full",
        "metrics": metrics,
        "queries": per_query_results,
    }


async def run_evaluation(
    queries: List[Dict[str, Any]],
    top_k: int = 5,
    model: str = "gpt-3.5-turbo",
) -> Dict[str, Any]:
    """Compatibility wrapper for existing callers."""
    return await run_full_evaluation(queries=queries, top_k=top_k, model=model)
