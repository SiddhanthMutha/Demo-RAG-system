"""
Abstract base parser interface for all document parsers.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List

from src.models import Document, DocumentType


class ParsingError(Exception):
    """Raised when document parsing fails."""

    def __init__(self, message: str, source: str = "", cause: Exception | None = None) -> None:
        super().__init__(message)
        self.source = source
        self.cause = cause

    def __str__(self) -> str:
        msg = super().__str__()
        if self.source:
            msg = f"[source={self.source}] {msg}"
        if self.cause:
            msg = f"{msg} | caused by: {self.cause}"
        return msg


class BaseParser(ABC):
    """Abstract base class all document parsers must implement."""

    @property
    @abstractmethod
    def supported_types(self) -> List[DocumentType]:
        """Return list of DocumentTypes this parser handles."""
        ...

    @abstractmethod
    async def parse(self, source: str, **kwargs: object) -> Document:
        """
        Parse a document from source into a Document object.

        Args:
            source: File path, URL, or raw content string.
            **kwargs: Parser-specific options.

        Returns:
            Document with extracted content and metadata.

        Raises:
            ParsingError: If parsing fails for any reason.
        """
        ...

    async def parse_batch(self, sources: List[str], **kwargs: object) -> List[Document | Exception]:
        """
        Parse multiple sources concurrently.

        Returns:
            List of Documents or exceptions (one per source).
        """
        tasks = [self.parse(source, **kwargs) for source in sources]
        return await asyncio.gather(*tasks, return_exceptions=True)  # type: ignore[return-value]
