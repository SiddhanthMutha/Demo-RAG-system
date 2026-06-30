"""
HTML view routes: server-rendered pages and HTMX partial endpoints.
"""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.eval import EvalRunRequest, execute_evaluation_run
from src.api.routes.ingest import ingest_upload, perform_ingestion
from src.database import get_session
from src.database.repository import DocumentRepository, EvaluationRepository
from src.models import ChunkingStrategy, DocumentType, IngestionRequest

router = APIRouter(tags=["Views"])

_templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


def _render(template_name: str, request: Request, **context: object):
    return templates.TemplateResponse(request, template_name, {"request": request, **context})


@router.get("/web/query", summary="Query page")
async def query_page(request: Request):
    return _render("query.html", request)


@router.post("/web/query/run", summary="Query session partial")
async def query_run_partial(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(5),
    model: str = Form("gpt-3.5-turbo"),
):
    stream_url = request.url_for("query_stream_sse").include_query_params(
        query=query,
        top_k=top_k,
        model=model,
    )
    return _render(
        "partials/query_session.html",
        request,
        session_id=str(uuid4()),
        stream_url=str(stream_url),
        query=query,
        model=model,
        top_k=top_k,
    )


@router.get("/web/ingest", summary="Ingest page")
async def ingest_page(request: Request):
    return _render("ingest.html", request)


@router.post("/web/ingest/source", summary="Ingest source partial")
async def ingest_source_partial(
    request: Request,
    source: str = Form(...),
    doc_type: str = Form(...),
    chunking_strategy: str = Form("semantic"),
    session: AsyncSession = Depends(get_session),
):
    try:
        payload = IngestionRequest(
            source=source,
            doc_type=DocumentType(doc_type),
            chunking_strategy=ChunkingStrategy(chunking_strategy),
        )
        result = await perform_ingestion(payload, session)
        return _render("partials/ingest_result.html", request, result=result)
    except Exception as exc:
        return _render("partials/error_state.html", request, message=str(exc))


@router.post("/web/ingest/upload", summary="Upload ingest partial")
async def ingest_upload_partial(
    request: Request,
    file: UploadFile = File(...),
    chunking_strategy: str = Form("semantic"),
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await ingest_upload(file=file, chunking_strategy=chunking_strategy, session=session)
        return _render("partials/ingest_result.html", request, result=result)
    except Exception as exc:
        return _render("partials/error_state.html", request, message=str(exc))


@router.get("/web/documents", summary="Documents page")
async def documents_page(request: Request):
    return _render("documents.html", request)


@router.get("/web/documents/table", summary="Documents table partial")
async def documents_table_partial(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    repo = DocumentRepository(session)
    documents = await repo.list_documents(limit=50)
    return _render("partials/documents_table.html", request, documents=documents)


@router.get("/web/eval", summary="Evaluation dashboard")
async def eval_page(request: Request):
    return _render("eval.html", request)


@router.get("/web/eval/history", summary="Evaluation history partial")
async def eval_history_partial(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    repo = EvaluationRepository(session)
    runs = await repo.list_grouped_runs(limit=10)
    return _render("partials/eval_history.html", request, runs=runs)


@router.get("/web/eval/run/{run_id}", summary="Evaluation run detail partial")
async def eval_run_detail_partial(
    request: Request,
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    repo = EvaluationRepository(session)
    run = await repo.get_run_details(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return _render("partials/eval_run_detail.html", request, run=run)


@router.post("/web/eval/run", summary="Run evaluation partial")
async def eval_run_partial(
    request: Request,
    top_k: int = Form(5),
    model: str = Form("gpt-3.5-turbo"),
    synthetic: bool = Form(True),
    max_qa_pairs: int = Form(20),
    sample_size: int = Form(20),
    session: AsyncSession = Depends(get_session),
):
    payload = EvalRunRequest(
        synthetic=synthetic,
        top_k=top_k,
        model=model,
        max_qa_pairs=max_qa_pairs,
        sample_size=sample_size,
    )
    try:
        result = await execute_evaluation_run(payload, session)
        repo = EvaluationRepository(session)
        runs = await repo.list_grouped_runs(limit=10)
        return _render(
            "partials/eval_run_response.html",
            request,
            result=result,
            runs=runs,
        )
    except Exception as exc:
        return _render("partials/error_state.html", request, message=str(exc))
