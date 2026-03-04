"""
Context window manager: fits retrieved chunks within model token budget.
"""
from typing import List, Tuple

import tiktoken
from loguru import logger

from src.models import RetrievalResult

# Context window sizes per model
MODEL_LIMITS: dict[str, int] = {
    "gpt-3.5-turbo": 16385,
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet": 200000,
}

DEFAULT_RESPONSE_BUDGET = 1000  # tokens reserved for the answer
DEFAULT_PROMPT_OVERHEAD = 500   # tokens for system prompt + question framing


class ContextManager:
    """
    Selects which retrieved chunks to include in the prompt based on
    token budget constraints.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name
        self.max_tokens = MODEL_LIMITS.get(model_name, 4096)
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Return token count for text."""
        return len(self.encoder.encode(text))

    def fit_context(
        self,
        chunks: List[RetrievalResult],
        system_prompt: str,
        query: str,
        max_response_tokens: int = DEFAULT_RESPONSE_BUDGET,
    ) -> Tuple[List[RetrievalResult], int]:
        """
        Select chunks that fit within the context window.

        Algorithm:
        1. Calculate fixed token cost (system prompt + query + response budget + overhead).
        2. Sort chunks by score descending (highest relevance first).
        3. Greedily add chunks until budget exhausted.

        Args:
            chunks: Candidate chunks (will be sorted by score internally).
            system_prompt: System instruction string.
            query: User query string.
            max_response_tokens: Tokens to reserve for the model's answer.

        Returns:
            Tuple of (selected chunks, remaining context token budget).
        """
        fixed_tokens = (
            self.count_tokens(system_prompt)
            + self.count_tokens(query)
            + max_response_tokens
            + DEFAULT_PROMPT_OVERHEAD
        )
        context_budget = self.max_tokens - fixed_tokens

        if context_budget <= 0:
            logger.warning(
                "No token budget left for context",
                model=self.model_name,
                fixed_tokens=fixed_tokens,
                max_tokens=self.max_tokens,
            )
            return [], 0

        # Sort by relevance score descending
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        selected: List[RetrievalResult] = []
        used_tokens = 0
        for chunk in sorted_chunks:
            # +20 per chunk for citation label and source line overhead
            chunk_tokens = self.count_tokens(chunk.content) + 20
            if used_tokens + chunk_tokens > context_budget:
                break
            selected.append(chunk)
            used_tokens += chunk_tokens

        remaining = context_budget - used_tokens
        logger.debug(
            "Context fitted",
            total_chunks=len(chunks),
            selected=len(selected),
            used_tokens=used_tokens,
            context_budget=context_budget,
        )
        return selected, remaining
