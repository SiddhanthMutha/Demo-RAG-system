"""
Query routes: synchronous query endpoint and WebSocket streaming endpoint.
"""
import json
import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_session
from src.database.repository import QueryRepository
from src.generation.context_manager import ContextManager
from src.generation.llm_client import LLMClient, LLMError
from src.generation.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from src.generation.streaming import StreamingHandler
from src.models import QueryRequest, QueryResponse, RetrievalResult
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/api/v1", tags=["Query"])

# Lazy singletons
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


# ------------------------------------------------------------------
# Core RAG pipeline logic (shared by sync and streaming endpoints)
# ------------------------------------------------------------------

async def _run_retrieval(query: str, top_k: int, filters: dict | None) -> list[RetrievalResult]:
    """Embed query and perform vector search."""
    emb_service = _get_embedding_service()
    query_embeddings = await emb_service.embed_batch([query])
    query_embedding = query_embeddings[0]

    vs = _get_vector_store()
    results = await vs.search(query_embedding=query_embedding, top_k=top_k * 2, filters=filters)
    return results


# ------------------------------------------------------------------
# Synchronous query endpoint
# ------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system (non-streaming)",
)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    """
    Query the RAG system and return a complete response.

    Steps:
    1. Embed query.
    2. Retrieve relevant chunks (vector search).
    3. Fit chunks into context window.
    4. Build RAG prompt.
    5. Generate response via LLM.
    6. Return answer with sources.
    """
    start = time.time()
    query_id = str(uuid4())

    # -- Retrieve --
    try:
        raw_results = await _run_retrieval(request.query, request.top_k, request.filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not raw_results:
        return QueryResponse(
            query_id=query_id,
            answer="I could not find any relevant information in the knowledge base.",
            sources=[],
        )

    # -- Context management --
    provider = "openai"
    model_name = request.model
    if request.model == "claude-3-5-sonnet":
        provider = "anthropic"
        model_name = "claude-3-5-sonnet-20241022"

    ctx_manager = ContextManager(model_name=model_name)
    selected_chunks, _ = ctx_manager.fit_context(
        chunks=raw_results,
        system_prompt=SYSTEM_PROMPT,
        query=request.query,
    )

    if not selected_chunks:
        selected_chunks = raw_results[: request.top_k]

    # -- Build prompt --
    prompt = build_rag_prompt(query=request.query, retrieved_chunks=selected_chunks)

    # -- Generate --
    llm = LLMClient(provider=provider, model=model_name)
    try:
        answer, input_tokens, output_tokens = await llm.generate(
            prompt=prompt, system_prompt=SYSTEM_PROMPT
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    cost_usd = llm.estimate_cost(input_tokens, output_tokens)
    latency_ms = round((time.time() - start) * 1000)

    # -- Log to DB --
    if settings.enable_query_logging:
        try:
            repo = QueryRepository(session)
            await repo.log_query(
                query_id=query_id,
                query_text=request.query,
                response_text=answer,
                retrieval_results=[
                    {"chunk_id": r.chunk_id, "score": r.score} for r in selected_chunks
                ],
                latency_ms=latency_ms,
                tokens_used=input_tokens + output_tokens,
                cost_usd=cost_usd,
                model_used=model_name,
            )
        except Exception as e:
            logger.warning("Query logging failed", error=str(e))

    logger.info(
        "Query processed",
        query_id=query_id,
        sources=len(selected_chunks),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )

    return QueryResponse(
        query_id=query_id,
        answer=answer,
        sources=selected_chunks,
        metadata={
            "tokens_used": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "model": model_name,
        },
    )


# ------------------------------------------------------------------
# WebSocket streaming endpoint
# ------------------------------------------------------------------

@router.websocket("/query/stream")
async def query_stream(websocket: WebSocket) -> None:
    """
    Streaming query endpoint over WebSocket.

    Client sends:  {"query": str, "top_k": int, "model": str}
    Server sends:
        {"type": "token",   "data": "<token>"}  — for each LLM token
        {"type": "sources", "data": [...]}        — final sources
        {"type": "done",    "data": null}          — stream complete
        {"type": "error",   "data": "<message>"}  — on failure
    """
    await websocket.accept()
    handler = StreamingHandler(websocket)

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)

        query_text = payload.get("query", "").strip()
        top_k = int(payload.get("top_k", settings.default_top_k))
        model_name = payload.get("model", "gpt-3.5-turbo")
        filters = payload.get("filters")

        if not query_text:
            await handler.send_error("Query must not be empty.")
            return

        # Retrieve
        raw_results = await _run_retrieval(query_text, top_k, filters)
        if not raw_results:
            await handler.send_token("I could not find relevant information in the knowledge base.")
            await handler.send_sources([])
            await handler.send_done()
            return

        # Context management
        provider = "openai"
        actual_model = model_name
        if model_name == "claude-3-5-sonnet":
            provider = "anthropic"
            actual_model = "claude-3-5-sonnet-20241022"

        ctx_manager = ContextManager(model_name=actual_model)
        selected_chunks, _ = ctx_manager.fit_context(
            chunks=raw_results, system_prompt=SYSTEM_PROMPT, query=query_text
        )
        if not selected_chunks:
            selected_chunks = raw_results[:top_k]

        # Build prompt
        prompt = build_rag_prompt(query=query_text, retrieved_chunks=selected_chunks)

        # Stream generation
        llm = LLMClient(provider=provider, model=actual_model)
        token_stream = llm.generate_stream(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        await handler.stream_tokens(token_stream)

        # Send sources and done signal
        await handler.send_sources(selected_chunks)
        await handler.send_done()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except json.JSONDecodeError:
        await handler.send_error("Invalid JSON payload.")
    except Exception as e:
        logger.error("WebSocket query error", error=str(e))
        await handler.send_error(f"Internal error: {e}")
