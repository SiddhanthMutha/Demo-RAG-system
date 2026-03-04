"""
Embedding service supporting local sentence-transformers and OpenAI embeddings.
Uses LRU cache for single-text lookups to avoid redundant computation.
"""
import asyncio
from functools import lru_cache
from typing import List, Optional

from loguru import logger

from src.config import settings


class EmbeddingService:
    """
    Generates vector embeddings for text.

    Supports:
    - Local: sentence-transformers/all-MiniLM-L6-v2  (384 dim, no API key)
    - OpenAI: text-embedding-3-small                  (1536 dim)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.use_openai = use_openai
        self._dimension: int

        if use_openai:
            import openai

            self._openai_model = "text-embedding-3-small"
            self._openai_client = openai.AsyncOpenAI(
                api_key=openai_api_key or settings.openai_api_key
            )
            self._dimension = 1536
            logger.info("EmbeddingService initialized with OpenAI", model=self._openai_model)
        else:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _model_name = model_name or settings.embedding_model
            self._local_model = SentenceTransformer(_model_name)
            self._dimension = self._local_model.get_sentence_embedding_dimension()  # type: ignore[assignment]
            logger.info("EmbeddingService initialized locally", model=_model_name, dim=self._dimension)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (synchronous).
        Result is cached in-process via LRU cache wrapper.
        """
        return self._embed_text_cached(text)

    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts to embed per API/model call.

        Returns:
            List of embedding vectors in the same order as texts.
        """
        if self.use_openai:
            return await self._openai_batch(texts, batch_size)
        else:
            return await self._local_batch(texts, batch_size)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @lru_cache(maxsize=2000)  # type: ignore[misc]
    def _embed_text_cached(self, text: str) -> List[float]:
        """Cache-backed single embedding."""
        if self.use_openai:
            raise RuntimeError(
                "Use embed_batch() for OpenAI embeddings (requires async context)"
            )
        embedding = self._local_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()  # type: ignore[union-attr]

    async def _local_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Run sentence-transformers in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        results: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: self._local_model.encode(  # type: ignore[misc]
                    b, normalize_embeddings=True, show_progress_bar=False
                ).tolist(),
            )
            results.extend(embeddings)

        logger.debug("Local batch embedded", count=len(texts))
        return results

    async def _openai_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Call OpenAI Embeddings API in batches with retry."""
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        async def _call_api(batch: List[str]) -> List[List[float]]:
            response = await self._openai_client.embeddings.create(
                model=self._openai_model, input=batch
            )
            return [item.embedding for item in response.data]

        results: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await _call_api(batch)
            results.extend(embeddings)

        logger.debug("OpenAI batch embedded", count=len(texts))
        return results
