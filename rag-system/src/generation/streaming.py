"""
Streaming response handler for WebSocket token-by-token delivery.
"""
import json
from typing import Any, AsyncIterator, Dict, List

from fastapi import WebSocket
from loguru import logger

from src.models import RetrievalResult


class StreamingHandler:
    """
    Manages server-side streaming over WebSocket connections.

    WebSocket message protocol:
    - {"type": "token",   "data": "<token_string>"}  — each LLM token
    - {"type": "sources", "data": [{...}, ...]}        — final source list
    - {"type": "done",    "data": null}                — stream complete
    - {"type": "error",   "data": "<message>"}         — error occurred
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    async def send_token(self, token: str) -> None:
        """Send a single token to the client."""
        await self._send({"type": "token", "data": token})

    async def send_sources(self, sources: List[RetrievalResult]) -> None:
        """Send the source list after generation completes."""
        sources_data = [
            {
                "chunk_id": s.chunk_id,
                "content": s.content[:300],  # Truncate for wire size
                "score": round(s.score, 4),
                "document_source": s.document_source,
                "metadata": s.metadata,
            }
            for s in sources
        ]
        await self._send({"type": "sources", "data": sources_data})

    async def send_done(self) -> None:
        """Signal stream completion."""
        await self._send({"type": "done", "data": None})

    async def send_error(self, message: str) -> None:
        """Send error message to client."""
        await self._send({"type": "error", "data": message})

    async def stream_tokens(self, token_stream: AsyncIterator[str]) -> str:
        """
        Stream all tokens to the WebSocket client.

        Returns:
            The full concatenated response text.
        """
        full_response = ""
        try:
            async for token in token_stream:
                await self.send_token(token)
                full_response += token
        except Exception as e:
            logger.error("Error during token streaming", error=str(e))
            await self.send_error(str(e))
        return full_response

    async def _send(self, payload: Dict[str, Any]) -> None:
        """Send JSON payload over WebSocket."""
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.warning("WebSocket send failed", error=str(e))
