"""
Document chunking strategies: fixed-size, semantic, and syntax-aware.
Uses tiktoken for accurate token counting.
"""
import re
from typing import List

import tiktoken
from loguru import logger

from src.models import Chunk, ChunkingStrategy, Document, DocumentType


class Chunker:
    """
    Splits Documents into Chunks using configurable strategies.

    Strategies:
    - FIXED_SIZE: Sliding window of max_tokens with overlap_tokens overlap.
    - SEMANTIC: Split on sentence boundaries, combine up to max_tokens.
    - SYNTAX_AWARE: One function/class per chunk (for code documents).
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        model_name: str = "gpt-3.5-turbo",
    ) -> None:
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # cl100k_base covers gpt-3.5-turbo / gpt-4; fallback for unknown models
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Split a Document into Chunks based on the configured strategy.

        Args:
            doc: Source document.

        Returns:
            Ordered list of Chunk objects (chunk_index 0, 1, 2…).
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            raw_chunks = self._fixed_size_chunk(doc.content)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            raw_chunks = self._semantic_chunk(doc.content)
        elif self.strategy == ChunkingStrategy.SYNTAX_AWARE:
            raw_chunks = self._syntax_aware_chunk(doc.content, doc.doc_type)
        else:
            raw_chunks = self._fixed_size_chunk(doc.content)

        chunks = []
        for idx, text in enumerate(raw_chunks):
            if not text.strip():
                continue
            token_count = self._count_tokens(text)
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    content=text,
                    token_count=token_count,
                    chunk_index=idx,
                    metadata={
                        "doc_type": doc.doc_type.value,
                        "source": doc.source,
                        "strategy": self.strategy.value,
                    },
                )
            )

        logger.debug(
            "Chunked document",
            doc_id=doc.id,
            strategy=self.strategy.value,
            num_chunks=len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Return token count for the given text."""
        return len(self.encoder.encode(text))

    def _encode(self, text: str) -> List[int]:
        return self.encoder.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fixed_size_chunk(self, text: str) -> List[str]:
        """
        Sliding window chunking: max_tokens per chunk, overlap_tokens overlap.
        Operates on token IDs for precision.
        """
        tokens = self._encode(text)
        chunks: List[str] = []

        step = self.max_tokens - self.overlap_tokens
        if step <= 0:
            step = self.max_tokens  # safety guard

        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._decode(chunk_tokens))
            if end == len(tokens):
                break
            start += step

        return chunks

    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Sentence-boundary chunking: accumulate sentences until max_tokens,
        then start a new chunk with overlap_tokens worth of sentences.
        """
        # Split on sentence boundaries
        sentences = self._split_sentences(text)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Single sentence exceeds max — force split it
            if sentence_tokens > self.max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # Fall back to fixed-size for this oversized sentence
                sub_chunks = self._fixed_size_chunk(sentence)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sentence_tokens > self.max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Keep last N sentences for overlap
                overlap_sentences = self._get_overlap_sentences(current_sentences)
                current_sentences = overlap_sentences
                current_tokens = sum(self._count_tokens(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _syntax_aware_chunk(self, text: str, doc_type: DocumentType) -> List[str]:
        """
        Code-aware chunking: attempt to keep functions/classes together.
        Falls back to fixed-size if parsing isn't available.
        """
        if doc_type in (DocumentType.CODE, DocumentType.MARKDOWN):
            try:
                return self._chunk_by_code_blocks(text)
            except Exception:
                pass
        return self._fixed_size_chunk(text)

    def _chunk_by_code_blocks(self, text: str) -> List[str]:
        """
        Split code by top-level function/class definitions.
        Uses simple regex heuristics; tree-sitter AST used in code_parser.py.
        """
        # Pattern: lines starting with 'def ', 'class ', 'async def '
        pattern = re.compile(r"^(?:(?:async\s+)?def |class )", re.MULTILINE)
        splits = [m.start() for m in pattern.finditer(text)]

        if not splits:
            return self._fixed_size_chunk(text)

        blocks: List[str] = []
        for i, start in enumerate(splits):
            end = splits[i + 1] if i + 1 < len(splits) else len(text)
            block = text[start:end].strip()
            if not block:
                continue
            # If block exceeds max, split further
            if self._count_tokens(block) > self.max_tokens:
                blocks.extend(self._fixed_size_chunk(block))
            else:
                blocks.append(block)

        return blocks

    def _split_sentences(self, text: str) -> List[str]:
        """Naive sentence splitter on '. ', '? ', '! ', and newlines."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split on sentence-ending punctuation followed by space/newline
        parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Return trailing sentences that fill overlap_tokens budget."""
        overlap: List[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            t = self._count_tokens(sentence)
            if tokens + t > self.overlap_tokens:
                break
            overlap.insert(0, sentence)
            tokens += t
        return overlap
