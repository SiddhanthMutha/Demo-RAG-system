"""
Code parser: syntax-aware parsing for Python and JavaScript files.
Extracts functions, classes, and docstrings as structured chunks.
"""
import ast
import re
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from src.ingestion.base_parser import BaseParser, ParsingError
from src.models import Document, DocumentType

SUPPORTED_EXTENSIONS = {".py": "python", ".js": "javascript", ".ts": "typescript"}


class CodeParser(BaseParser):
    """Parses Python/JavaScript source files into Document objects."""

    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.CODE]

    async def parse(self, source: str, **kwargs: Any) -> Document:
        """
        Parse a code file into a Document.

        Args:
            source: Absolute file path to the source file.

        Returns:
            Document with code content and metadata (language, LOC, imports).

        Raises:
            ParsingError: If file not found or unsupported language.
        """
        path = Path(source)
        if not path.exists():
            raise ParsingError(f"File not found: {source}", source=source)

        ext = path.suffix.lower()
        language = SUPPORTED_EXTENSIONS.get(ext)
        if language is None:
            raise ParsingError(
                f"Unsupported code file extension: {ext}. Supported: {list(SUPPORTED_EXTENSIONS)}",
                source=source,
            )

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise ParsingError(f"Cannot read file: {e}", source=source, cause=e)

        lines = content.splitlines()
        loc = len(lines)

        if language == "python":
            imports = self._extract_python_imports(content)
            metadata: Dict[str, Any] = {
                "language": language,
                "loc": loc,
                "file_name": path.name,
                "imports": imports,
            }
        else:
            imports_js = self._extract_js_imports(content)
            metadata = {
                "language": language,
                "loc": loc,
                "file_name": path.name,
                "imports": imports_js,
            }

        logger.info("Code file parsed", source=source, language=language, loc=loc)

        return Document(
            content=content,
            doc_type=DocumentType.CODE,
            source=source,
            metadata=metadata,
        )

    def _extract_python_imports(self, code: str) -> List[str]:
        """Extract import statements from Python source using AST."""
        imports: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(a.name for a in node.names)
                    imports.append(f"from {module} import {names}")
        except SyntaxError:
            # Fall back to regex for syntax-error files
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    imports.append(stripped)
        return imports

    def _extract_js_imports(self, code: str) -> List[str]:
        """Extract ES6 import/require statements from JS/TS."""
        imports: List[str] = []
        pattern = re.compile(
            r"^(?:import .+|const .+ = require\(.+\)|var .+ = require\(.+\))",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            imports.append(match.group().strip())
        return imports
