"""
Document ingestion routes: file upload, URL, and batch ingestion.
"""
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_session
from src.database.repository import DocumentRepository
from src.ingestion.chunker import Chunker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.base_parser import ParsingError
from src.models import (
    ChunkingStrategy,
    DocumentType,
    IngestionRequest,
    IngestionResponse,
)
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

# Lazy-initialized singletons (created once per process)
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStore] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a document by file path or URL",
)
async def ingest_document(
    request: IngestionRequest,
    session: AsyncSession = Depends(get_session),
) -> IngestionResponse:
    return await perform_ingestion(request, session)


async def perform_ingestion(
    request: IngestionRequest,
    session: AsyncSession,
) -> IngestionResponse:
    """
    Ingest a document into the RAG system.

    Steps:
    1. Parse document based on type.
    2. Check for duplicates (content hash).
    3. Chunk content using chosen strategy.
    4. Generate embeddings for all chunks.
    5. Store chunks in vector DB (Pinecone) and metadata in PostgreSQL.
    """
    start = time.time()
    errors: list[str] = []

    # -- Parse --
    try:
        parser = _get_parser(request.doc_type)
        doc = await parser.parse(request.source)
        if request.metadata:
            doc.metadata.update(request.metadata)
    except ParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")

    # -- Deduplication --
    repo = DocumentRepository(session)
    existing_id = await repo.document_exists(doc.content)
    if existing_id:
        logger.info("Duplicate document skipped", existing_id=existing_id)
        return IngestionResponse(
            document_id=existing_id,
            chunks_created=0,
            status="success",
            errors=["Document already exists (content hash match)."],
        )

    # -- Chunk --
    chunker = Chunker(
        strategy=request.chunking_strategy,
        max_tokens=settings.default_chunk_size,
        overlap_tokens=settings.default_chunk_overlap,
    )
    chunks = chunker.chunk_document(doc)

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks.")

    # -- Embed --
    emb_service = get_embedding_service()
    texts = [c.content for c in chunks]
    try:
        embeddings = await emb_service.embed_batch(texts)
    except Exception as e:
        errors.append(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    # -- Persist to PostgreSQL --
    try:
        await repo.save_document(doc)
        await repo.save_chunks(chunks)
    except Exception as e:
        logger.warning("DB persistence failed (proceeding with vector store)", error=str(e))
        errors.append(f"DB storage warning: {e}")

    # -- Persist to Vector Store --
    vs = get_vector_store()
    try:
        await vs.upsert_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store upsert failed: {e}")

    elapsed = round((time.time() - start) * 1000)
    logger.info(
        "Document ingested",
        doc_id=doc.id,
        source=request.source,
        chunks=len(chunks),
        elapsed_ms=elapsed,
    )

    return IngestionResponse(
        document_id=doc.id,
        chunks_created=len(chunks),
        status="success" if not errors else "partial",
        errors=errors,
    )


@router.post(
    "/ingest/upload",
    response_model=IngestionResponse,
    summary="Upload and ingest a PDF file",
)
async def ingest_upload(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(default="semantic"),
    session: AsyncSession = Depends(get_session),
) -> IngestionResponse:
    """Upload a PDF file directly and ingest it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported for upload.")

    # Save to temp location in data/raw/
    import shutil
    import tempfile

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", dir="data/raw", prefix="upload_"
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        strategy = ChunkingStrategy(chunking_strategy)
    except ValueError:
        strategy = ChunkingStrategy.SEMANTIC

    request = IngestionRequest(
        source=tmp_path,
        doc_type=DocumentType.PDF,
        chunking_strategy=strategy,
    )
    return await ingest_document(request, session)


def _get_parser(doc_type: DocumentType):  # type: ignore[return]
    """Return the appropriate parser for the document type."""
    from src.ingestion.web_parser import WebParser
    from src.ingestion.code_parser import CodeParser
    from src.ingestion.markdown_parser import MarkdownParser

    parsers = {
        DocumentType.PDF: PDFParser,
        DocumentType.WEB: WebParser,
        DocumentType.CODE: CodeParser,
        DocumentType.MARKDOWN: MarkdownParser,
    }
    parser_class = parsers.get(doc_type)
    if parser_class is None:
        raise HTTPException(
            status_code=501,
            detail=f"No parser implemented for doc_type='{doc_type.value}'.",
        )
    return parser_class()
