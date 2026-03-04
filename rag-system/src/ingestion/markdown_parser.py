"""
Markdown parser: converts Markdown files into Documents with section-aware metadata.
"""
from pathlib import Path
from typing import Any, Dict, List
import re

from loguru import logger

from src.ingestion.base_parser import BaseParser, ParsingError
from src.models import Document, DocumentType


class MarkdownParser(BaseParser):
    """Parses .md and .markdown files into Document objects."""

    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.MARKDOWN]

    async def parse(self, source: str, **kwargs: Any) -> Document:
        """
        Parse a Markdown file.

        Args:
            source: Absolute path to the .md file.

        Returns:
            Document with markdown text and metadata (title, section count, headings).

        Raises:
            ParsingError: If file is missing or unreadable.
        """
        path = Path(source)
        if not path.exists():
            raise ParsingError(f"File not found: {source}", source=source)
        if path.suffix.lower() not in (".md", ".markdown", ".mdx"):
            raise ParsingError(f"Not a markdown file: {source}", source=source)

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise ParsingError(f"Cannot read file: {e}", source=source, cause=e)

        if not content.strip():
            raise ParsingError(f"Empty markdown file: {source}", source=source)

        metadata = self._extract_metadata(content, path)
        # Strip frontmatter from content before storing
        clean_content = self._strip_frontmatter(content)

        logger.info(
            "Markdown parsed",
            source=source,
            title=metadata.get("title", ""),
            sections=metadata.get("section_count", 0),
        )

        return Document(
            content=clean_content,
            doc_type=DocumentType.MARKDOWN,
            source=source,
            metadata=metadata,
        )

    def _extract_metadata(self, content: str, path: Path) -> Dict[str, Any]:
        """Extract title, headings, and YAML frontmatter from markdown."""
        metadata: Dict[str, Any] = {"file_name": path.name}

        # YAML frontmatter
        frontmatter = self._parse_frontmatter(content)
        metadata.update(frontmatter)

        # Heading analysis
        headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
        if headings and "title" not in metadata:
            metadata["title"] = headings[0]  # First heading is likely the title
        metadata["section_count"] = len(headings)
        metadata["headings"] = headings[:10]  # Store first 10 headings

        return metadata

    def _parse_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract key: value pairs from YAML frontmatter (--- blocks)."""
        frontmatter: Dict[str, Any] = {}
        match = re.match(r"^---\s*\n(.+?)\n---\s*\n", content, re.DOTALL)
        if match:
            for line in match.group(1).splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    frontmatter[key.strip()] = value.strip().strip('"\'')
        return frontmatter

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter block from content."""
        return re.sub(r"^---\s*\n.+?\n---\s*\n", "", content, flags=re.DOTALL).strip()
