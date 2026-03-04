"""
Pytest fixtures shared across all test suites.
"""
import asyncio
import pytest
from typing import AsyncGenerator

from httpx import ASGITransport, AsyncClient

# -- Event loop (session-scoped) --

@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# -- Test HTTP client --

@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client pointing at the FastAPI app."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# -- Sample data --

@pytest.fixture
def sample_text() -> str:
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without being explicitly programmed. "
        "Deep learning uses neural networks with many layers. "
        "Natural language processing allows machines to understand human language. "
        "Transformers are a key architecture in modern NLP. "
        "BERT and GPT are well-known transformer-based models. "
    ) * 10  # Repeat to get enough content for chunking tests


@pytest.fixture
def sample_document(sample_text):
    from src.models import Document, DocumentType
    return Document(
        content=sample_text,
        doc_type=DocumentType.TEXT,
        source="test/sample.txt",
        metadata={"test": True},
    )


@pytest.fixture
def sample_chunks(sample_document):
    from src.ingestion.chunker import Chunker, ChunkingStrategy
    chunker = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, max_tokens=50, overlap_tokens=10)
    return chunker.chunk_document(sample_document)
