"""
LLM client supporting OpenAI and Anthropic with streaming and cost tracking.
"""
from typing import AsyncIterator, Dict, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# Pricing as of early 2025 (per 1K tokens)
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
}


class LLMError(Exception):
    """Raised when LLM API call fails after retries."""

    pass


class LLMClient:
    """
    Unified LLM client for OpenAI and Anthropic.

    Supports:
    - Streaming generation (yields token strings)
    - Non-streaming generation (returns full response)
    - Cost estimation
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model

        if self.provider == "openai":
            import openai
            from src.config import settings

            self._client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        elif self.provider == "anthropic":
            import anthropic
            from src.config import settings

            self._client = anthropic.AsyncAnthropic(  # type: ignore[assignment]
                api_key=api_key or settings.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'anthropic'.")

        logger.info("LLMClient initialized", provider=self.provider, model=self.model)

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """
        Yield response tokens as they are generated.

        Args:
            prompt: User message content.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum response tokens.

        Yields:
            String chunks/tokens.
        """
        if self.provider == "openai":
            async for token in self._openai_stream(prompt, system_prompt, temperature, max_tokens):
                yield token
        elif self.provider == "anthropic":
            async for token in self._anthropic_stream(prompt, system_prompt, temperature, max_tokens):
                yield token

    # ------------------------------------------------------------------
    # Non-streaming generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> tuple[str, int, int]:
        """
        Generate a complete response.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
        """
        if self.provider == "openai":
            return await self._openai_complete(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            return await self._anthropic_complete(prompt, system_prompt, temperature, max_tokens)
        raise LLMError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated cost in USD."""
        pricing = PRICING.get(self.model, {"input": 0.001, "output": 0.002})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000.0

    # ------------------------------------------------------------------
    # OpenAI internals
    # ------------------------------------------------------------------

    async def _openai_stream(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        import openai

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI streaming failed: {e}") from e

    async def _openai_complete(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, int, int]:
        import openai

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            text = response.choices[0].message.content or ""
            usage = response.usage
            return text, usage.prompt_tokens, usage.completion_tokens  # type: ignore[union-attr]
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI completion failed: {e}") from e

    # ------------------------------------------------------------------
    # Anthropic internals
    # ------------------------------------------------------------------

    async def _anthropic_stream(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(  # type: ignore[union-attr]
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise LLMError(f"Anthropic streaming failed: {e}") from e

    async def _anthropic_complete(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, int, int]:
        try:
            response = await self._client.messages.create(  # type: ignore[union-attr]
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.content[0].text if response.content else ""
            return text, response.usage.input_tokens, response.usage.output_tokens
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {e}") from e
