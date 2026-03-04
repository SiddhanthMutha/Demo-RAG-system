"""
PDF document parser using PyPDF2 with pdfplumber fallback.
Extracts text, page numbers, and document metadata.
"""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.ingestion.base_parser import BaseParser, ParsingError
from src.models import Document, DocumentType


class PDFParser(BaseParser):
    """Parses PDF files into Document objects."""

    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.PDF]

    async def parse(self, source: str, password: Optional[str] = None, **kwargs: Any) -> Document:
        """
        Extract text from a PDF file.

        Args:
            source: Absolute file path to the PDF.
            password: Optional password for protected PDFs.

        Returns:
            Document with extracted text and metadata.

        Raises:
            ParsingError: If the file is missing, corrupt, or unreadable.
        """
        path = Path(source)
        if not path.exists():
            raise ParsingError(f"File not found: {source}", source=source)
        if not path.suffix.lower() == ".pdf":
            raise ParsingError(f"Not a PDF file: {source}", source=source)

        logger.info("Parsing PDF", source=source)

        try:
            text, metadata = await self._extract_with_pypdf2(path, password)
        except Exception as primary_err:
            logger.warning(
                "PyPDF2 failed, falling back to pdfplumber",
                source=source,
                error=str(primary_err),
            )
            try:
                text, metadata = await self._extract_with_pdfplumber(path, password)
            except Exception as fallback_err:
                raise ParsingError(
                    f"All parsers failed for {source}",
                    source=source,
                    cause=fallback_err,
                ) from fallback_err

        if not text.strip():
            raise ParsingError(f"No extractable text found in {source}", source=source)

        return Document(
            content=text,
            doc_type=DocumentType.PDF,
            source=source,
            metadata={**metadata, "file_name": path.name, "file_size_bytes": path.stat().st_size},
        )

    async def _extract_with_pypdf2(
        self, path: Path, password: Optional[str]
    ) -> tuple[str, Dict[str, Any]]:
        """Primary extraction using PyPDF2."""
        import PyPDF2  # type: ignore

        pages_text: List[str] = []
        metadata: Dict[str, Any] = {}

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            # Handle password-protected PDFs
            if reader.is_encrypted:
                if password is None:
                    raise ParsingError(f"PDF is encrypted and no password provided: {path}")
                if not reader.decrypt(password):
                    raise ParsingError(f"Wrong password for PDF: {path}")

            # Extract document-level metadata
            info = reader.metadata
            if info:
                metadata = {
                    "title": info.get("/Title", ""),
                    "author": info.get("/Author", ""),
                    "subject": info.get("/Subject", ""),
                    "creator": info.get("/Creator", ""),
                    "page_count": len(reader.pages),
                }

            # Extract text page by page
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(f"[Page {page_num}]\n{page_text}")
                except Exception as e:
                    logger.warning("Failed to extract page", page=page_num, error=str(e))

        return "\n\n".join(pages_text), metadata

    async def _extract_with_pdfplumber(
        self, path: Path, password: Optional[str]
    ) -> tuple[str, Dict[str, Any]]:
        """Fallback extraction using pdfplumber (better for complex layouts)."""
        import pdfplumber  # type: ignore

        pages_text: List[str] = []
        metadata: Dict[str, Any] = {}

        open_kwargs = {"password": password} if password else {}
        with pdfplumber.open(str(path), **open_kwargs) as pdf:
            metadata = {
                "page_count": len(pdf.pages),
                "title": pdf.metadata.get("Title", "") if pdf.metadata else "",
                "author": pdf.metadata.get("Author", "") if pdf.metadata else "",
            }
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(f"[Page {page_num}]\n{page_text}")

        return "\n\n".join(pages_text), metadata
