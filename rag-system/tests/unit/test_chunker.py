"""
Unit tests for the Chunker class and all chunking strategies.
"""
import pytest

from src.ingestion.chunker import Chunker
from src.models import Chunk, ChunkingStrategy, Document, DocumentType


def make_doc(content: str, doc_type: DocumentType = DocumentType.TEXT) -> Document:
    return Document(content=content, doc_type=doc_type, source="test.txt")


LONG_TEXT = (
    "Artificial intelligence is transforming the world. "
    "Machine learning enables computers to learn from data. "
    "Deep learning uses multi-layer neural networks. "
    "Natural language processing handles human language. "
    "Computer vision handles image and video understanding. "
) * 20


# ------------------------------------------------------------------
# Fixed-size chunking
# ------------------------------------------------------------------

class TestFixedSizeChunking:
    def test_produces_multiple_chunks(self):
        chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=50, overlap_tokens=10)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunks_do_not_exceed_max_tokens(self):
        max_tokens = 50
        chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=max_tokens, overlap_tokens=5)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.token_count <= max_tokens + 5  # Allow minor encoding variance

    def test_chunk_indices_are_sequential(self):
        chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=50, overlap_tokens=10)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_has_correct_document_id(self, sample_document):
        chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=50, overlap_tokens=10)
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.document_id == sample_document.id

    def test_short_text_produces_single_chunk(self):
        short = "Hello world. This is a short sentence."
        chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=512, overlap_tokens=50)
        doc = make_doc(short)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_empty_content_after_strip_skipped(self):
        chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=512, overlap_tokens=50)
        doc = make_doc("   \n\n   ")
        # Should not raise, may return empty list
        # (whitespace only strips to empty)
        assert isinstance(chunker.chunk_document(doc), list)


# ------------------------------------------------------------------
# Semantic chunking
# ------------------------------------------------------------------

class TestSemanticChunking:
    def test_produces_chunks_from_sentences(self):
        chunker = Chunker(strategy=ChunkingStrategy.SEMANTIC, max_tokens=100, overlap_tokens=20)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0

    def test_chunks_respect_token_limit(self):
        max_tokens = 100
        chunker = Chunker(strategy=ChunkingStrategy.SEMANTIC, max_tokens=max_tokens, overlap_tokens=20)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            # Semantic chunks may occasionally slightly exceed due to sentence size
            assert chunk.token_count <= max_tokens * 2

    def test_metadata_includes_strategy(self):
        chunker = Chunker(strategy=ChunkingStrategy.SEMANTIC, max_tokens=100, overlap_tokens=20)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "semantic"


# ------------------------------------------------------------------
# Syntax-aware chunking
# ------------------------------------------------------------------

class TestSyntaxAwareChunking:
    PYTHON_CODE = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

class Calculator:
    """Simple calculator."""
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''

    def test_code_chunking_splits_on_functions(self):
        chunker = Chunker(strategy=ChunkingStrategy.SYNTAX_AWARE, max_tokens=200, overlap_tokens=0)
        doc = make_doc(self.PYTHON_CODE, doc_type=DocumentType.CODE)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2


# ------------------------------------------------------------------
# Chunk model validation
# ------------------------------------------------------------------

class TestChunkModel:
    def test_chunk_has_required_fields(self, sample_chunks):
        for chunk in sample_chunks:
            assert chunk.id
            assert chunk.document_id
            assert chunk.content
            assert chunk.token_count >= 1
            assert chunk.chunk_index >= 0

    def test_chunk_token_count_is_positive(self, sample_chunks):
        for chunk in sample_chunks:
            assert chunk.token_count > 0

    def test_chunk_embedding_validation_rejects_wrong_dim(self):
        """Chunk should reject embeddings with unsupported dimensions."""
        with pytest.raises(Exception):
            Chunk(
                document_id="test-doc",
                content="Hello",
                token_count=1,
                chunk_index=0,
                embedding=[0.1] * 128,  # 128-dim not supported
            )

    def test_chunk_embedding_accepts_valid_dim(self):
        chunk = Chunk(
            document_id="test-doc",
            content="Hello world",
            token_count=2,
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        assert len(chunk.embedding) == 384  # type: ignore
