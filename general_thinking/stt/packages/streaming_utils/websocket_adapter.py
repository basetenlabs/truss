"""
WebSocket Adapter for FastAPI/Starlette WebSocket.

Wraps a FastAPI WebSocket to match the truss_chains.WebSocketProtocol interface,
allowing StreamProcessor to work with both chain and truss WebSocket types.

Chain uses: truss_chains.WebSocketProtocol
  - receive() -> str | bytes
  - send_json(data) -> None
  - receive_text() -> str

Truss uses: FastAPI WebSocket
  - receive() -> dict ({"type": ..., "text": ..., "bytes": ...})
  - send_json(data) -> None
  - receive_text() -> str

This adapter bridges the gap by wrapping FastAPI WebSocket to match WebSocketProtocol.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


class FastAPIWebSocketAdapter:
    """
    Adapts FastAPI/Starlette WebSocket to match WebSocketProtocol interface.

    Usage:
        # In truss model.py websocket handler:
        adapter = FastAPIWebSocketAdapter(fastapi_ws)
        stream_processor = StreamProcessor(adapter, stream_id, ...)
    """

    def __init__(self, ws):
        """
        Args:
            ws: A FastAPI/Starlette WebSocket instance
        """
        self._ws = ws

    async def receive(self) -> Union[str, bytes]:
        """
        Receive a message, returning str or bytes.

        Converts FastAPI WebSocket's dict-based receive() to match
        WebSocketProtocol's simpler str | bytes return type.
        """
        message = await self._ws.receive()

        # Check for disconnect
        if message.get("type") == "websocket.disconnect":
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect(
                code=message.get("code", 1000),
                reason=message.get("reason", ""),
            )

        # Return bytes if present, otherwise text
        if "bytes" in message and message["bytes"] is not None:
            return message["bytes"]
        if "text" in message and message["text"] is not None:
            return message["text"]

        # Unexpected message format
        from fastapi import WebSocketDisconnect

        raise WebSocketDisconnect(code=1000, reason="Unexpected message format")

    async def send_json(self, data: dict) -> None:
        """Send JSON data (pass-through to FastAPI WebSocket)."""
        await self._ws.send_json(data)

    async def receive_text(self) -> str:
        """Receive text data (pass-through to FastAPI WebSocket)."""
        return await self._ws.receive_text()
