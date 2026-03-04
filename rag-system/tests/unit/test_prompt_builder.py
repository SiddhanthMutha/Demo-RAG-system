"""
Unit tests for prompt builder and context manager.
"""
import pytest

from src.generation.context_manager import ContextManager
from src.generation.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from src.models import RetrievalResult


def make_result(idx: int, content: str = None, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk_{idx}",
        content=content or f"This is chunk content number {idx}. " * 10,
        score=score,
        metadata={"doc_type": "text"},
        document_source=f"doc_{idx}.pdf",
    )


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

class TestPromptBuilder:
    def test_prompt_contains_query(self):
        query = "What is machine learning?"
        chunks = [make_result(1)]
        prompt = build_rag_prompt(query, chunks)
        assert query in prompt

    def test_prompt_contains_numbered_citations(self):
        chunks = [make_result(1), make_result(2), make_result(3)]
        prompt = build_rag_prompt("Test query", chunks)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_prompt_contains_source_references(self):
        chunks = [make_result(1)]
        prompt = build_rag_prompt("Test", chunks)
        assert "doc_1.pdf" in prompt

    def test_prompt_has_answer_section(self):
        chunks = [make_result(1)]
        prompt = build_rag_prompt("Test", chunks)
        assert "ANSWER:" in prompt

    def test_prompt_has_context_section(self):
        chunks = [make_result(1)]
        prompt = build_rag_prompt("Test", chunks)
        assert "CONTEXT:" in prompt

    def test_empty_chunks_produces_empty_context(self):
        prompt = build_rag_prompt("Test", [])
        # Should still have CONTEXT and QUESTION sections
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------

class TestContextManager:
    def test_fit_context_returns_chunks_within_budget(self):
        cm = ContextManager(model_name="gpt-3.5-turbo")
        chunks = [make_result(i, score=1.0 - i * 0.1) for i in range(20)]
        selected, remaining = cm.fit_context(chunks, SYSTEM_PROMPT, "What is AI?")
        assert len(selected) > 0
        assert remaining >= 0

    def test_fit_context_sorts_by_score(self):
        cm = ContextManager(model_name="gpt-3.5-turbo")
        chunks = [
            make_result(1, score=0.5),
            make_result(2, score=0.9),
            make_result(3, score=0.1),
        ]
        selected, _ = cm.fit_context(chunks, SYSTEM_PROMPT, "Test")
        if len(selected) >= 2:
            # Highest scores should appear first
            assert selected[0].score >= selected[1].score

    def test_count_tokens_positive(self):
        cm = ContextManager()
        tokens = cm.count_tokens("Hello, world! This is a test sentence.")
        assert tokens > 0

    def test_count_tokens_scales_with_length(self):
        cm = ContextManager()
        short = cm.count_tokens("Hello")
        long = cm.count_tokens("Hello " * 100)
        assert long > short

    def test_gpt4_has_smaller_context_than_claude(self):
        cm_gpt4 = ContextManager(model_name="gpt-4")
        cm_claude = ContextManager(model_name="claude-3-5-sonnet-20241022")
        assert cm_gpt4.max_tokens < cm_claude.max_tokens
