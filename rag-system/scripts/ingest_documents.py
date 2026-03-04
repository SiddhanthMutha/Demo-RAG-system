#!/usr/bin/env python3
"""
Bulk document ingestion script.
Ingests all documents from the data/raw/ directory.

Usage:
    python scripts/ingest_documents.py [--dir data/raw] [--type pdf]
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingestion.chunker import Chunker, ChunkingStrategy
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.web_parser import WebParser
from src.ingestion.code_parser import CodeParser
from src.ingestion.markdown_parser import MarkdownParser
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.models import DocumentType


EXTENSION_TO_TYPE = {
    ".pdf": (DocumentType.PDF, PDFParser),
    ".md": (DocumentType.MARKDOWN, MarkdownParser),
    ".markdown": (DocumentType.MARKDOWN, MarkdownParser),
    ".py": (DocumentType.CODE, CodeParser),
    ".js": (DocumentType.CODE, CodeParser),
    ".ts": (DocumentType.CODE, CodeParser),
}


async def ingest_file(path: Path, emb_service: EmbeddingService, vs: VectorStore) -> bool:
    ext = path.suffix.lower()
    if ext not in EXTENSION_TO_TYPE:
        logger.warning("Skipping unsupported file", path=str(path))
        return False

    doc_type, parser_class = EXTENSION_TO_TYPE[ext]
    parser = parser_class()
    chunker = Chunker(strategy=ChunkingStrategy.SEMANTIC)

    try:
        doc = await parser.parse(str(path))
        chunks = chunker.chunk_document(doc)
        embeddings = await emb_service.embed_batch([c.content for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        await vs.upsert_chunks(chunks)
        logger.info("✅ Ingested", path=path.name, chunks=len(chunks))
        return True
    except Exception as e:
        logger.error("❌ Failed", path=path.name, error=str(e))
        return False


async def main(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    if not directory.exists():
        logger.error("Directory not found", dir=str(directory))
        sys.exit(1)

    files = list(directory.iterdir())
    if args.type:
        files = [f for f in files if f.suffix.lower() == f".{args.type}"]

    logger.info("Starting bulk ingestion", directory=str(directory), file_count=len(files))

    emb_service = EmbeddingService()
    vs = VectorStore()

    results = await asyncio.gather(*[ingest_file(f, emb_service, vs) for f in files])
    success = sum(results)
    logger.info("Ingestion complete", success=success, total=len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk document ingestion")
    parser.add_argument("--dir", default="data/raw", help="Directory to scan")
    parser.add_argument("--type", help="File extension to filter (e.g. pdf)")
    asyncio.run(main(parser.parse_args()))
