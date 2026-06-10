import json
import logging
import time
import traceback
from typing import Any, Dict

from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from websockets.exceptions import ConnectionClosedError

from .error_utils import StreamError, WebSocketError, log_stream_event
from .message_types import Message, MessageType

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket communication with comprehensive error handling and logging."""

    def __init__(
        self, websocket: Any, stream_id: str
    ):  # WebSocketProtocol or FastAPIWebSocketAdapter
        self.websocket = websocket
        self.stream_id = stream_id
        self.connection_start_time = time.time()
        self.messages_sent = 0
        self.error_sent = 0
        self.messages_received = 0
        self.last_activity = time.time()
        self._closed = False

        log_stream_event(
            stream_id,
            "WebSocket manager initialized",
            {"connection_start_time": self.connection_start_time},
        )

    def mark_closed(self) -> None:
        """Mark the connection as closed."""
        if not self._closed:
            self._closed = True
            logger.debug(f"🔌 WebSocket connection marked as closed for stream {self.stream_id}")

    def is_connected(self) -> bool:
        """Check if WebSocket connection is still active."""
        if self._closed:
            return False

        try:
            # Use getattr to safely access attributes that may not be in the Protocol definition
            # Type checker doesn't know these exist on the Protocol, but they exist on Starlette's WebSocket
            client_state_attr = getattr(self.websocket, "client_state", None)  # type: ignore[attr-defined]
            app_state_attr = getattr(self.websocket, "application_state", None)  # type: ignore[attr-defined]

            if client_state_attr is not None and app_state_attr is not None:
                from starlette.websockets import WebSocketState

                # Access .value attribute on the WebSocketState enum
                client_state = client_state_attr.value  # type: ignore[attr-defined]
                app_state = app_state_attr.value  # type: ignore[attr-defined]

                # WebSocket is only truly connected when both states are CONNECTED
                is_connected = (
                    client_state == WebSocketState.CONNECTED.value
                    and app_state == WebSocketState.CONNECTED.value
                )

                if is_connected:
                    self.last_activity = time.time()
                else:
                    # Connection is not in CONNECTED state, mark as closed
                    self._closed = True
                    logger.debug(
                        f"🔍 WebSocket {self.stream_id} not connected: "
                        f"client_state={client_state}, app_state={app_state}"
                    )

                return is_connected
        except Exception as e:
            logger.debug(f"🔍 Error checking WebSocket state for stream {self.stream_id}: {e}")
            # If we can't check the state, assume disconnected if we've marked it closed
            return not self._closed

        # If we can't check the state, rely on our closed flag
        return not self._closed

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for monitoring."""
        return {
            "stream_id": self.stream_id,
            "connection_duration": time.time() - self.connection_start_time,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "last_activity": self.last_activity,
            "is_connected": self.is_connected(),
        }

    async def send_json_safe(self, payload: Dict[str, Any], is_error: bool = False) -> None:
        """
        Send JSON payload with comprehensive error handling and logging.

        Raises WebSocketError only for truly unexpected errors (not connection-closed cases).
        Connection-closed cases are handled gracefully and logged.
        """
        start_time = time.time()

        # Early return if connection is already closed
        if not self.is_connected():
            logger.debug(
                f"⚠️ WebSocket for stream {self.stream_id} is not connected, skipping send"
            )
            return

        try:
            await self.websocket.send_json(payload)
            if is_error:
                self.error_sent += 1
            else:
                self.messages_sent += 1
            self.last_activity = time.time()

            duration = time.time() - start_time

            payload_type = payload.get("type", "unknown")
            if payload_type == "transcription":
                suffix = "_final" if payload.get("is_final") else "_partial"
                payload_type = f"{payload_type}{suffix}"

            log_stream_event(
                self.stream_id,
                "Message sent successfully",
                {
                    "payload_type": payload_type,
                    "duration_ms": round(duration * 1000, 2),
                    "message_size": len(json.dumps(payload)),
                },
                "INFO",
            )
            return

        except (WebSocketDisconnect, StarletteWebSocketDisconnect, ConnectionClosedError) as e:
            duration = time.time() - start_time
            self._closed = True  # Mark connection as closed
            logger.warning(
                f"🔌 Client disconnected during send for stream {self.stream_id} after {duration:.3f}s: {e}"
            )
            return  # Don't raise - connection closed is expected

        except RuntimeError as e:
            # Starlette/ASGI raises RuntimeError for connection-closed cases with specific messages:
            # - "Cannot call "send" once a close message has been sent." (Starlette websockets.py:97)
            # - "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed." (Uvicorn)
            duration = time.time() - start_time
            error_str = str(e).lower()
            if any(
                keyword in error_str
                for keyword in [
                    "close message has been sent",  # Starlette: line 97
                    "websocket.close",  # Uvicorn: part of error message
                    "response already completed",  # Uvicorn: part of error message
                ]
            ):
                self._closed = True  # Mark connection as closed
                logger.warning(
                    f"🔌 Connection closed error during send for stream {self.stream_id} after {duration:.3f}s: {e}"
                )
                return  # Don't raise - connection closed is expected
            else:
                # RuntimeError for other reasons - re-raise as WebSocketError
                logger.error(
                    f"❌ RuntimeError sending to WebSocket for stream {self.stream_id} after {duration:.3f}s: {e}"
                )
                traceback.print_exc()
                raise WebSocketError(self.stream_id, f"Send operation failed: {str(e)}")

        except Exception as e:
            # Raise if truly unexpected error
            duration = time.time() - start_time
            logger.error(
                f"❌ Unexpected error sending to WebSocket for stream {self.stream_id} after {duration:.3f}s: {e}"
            )
            traceback.print_exc()
            raise WebSocketError(self.stream_id, f"Send operation failed: {str(e)}")

    def log_connection_stats(self) -> None:
        """Log current connection statistics."""
        stats = self.get_connection_stats()
        log_stream_event(self.stream_id, "Connection statistics", stats, "DEBUG")

    async def send_error_to_websocket(self, error: StreamError) -> None:
        """
        Send error message to WebSocket client with detailed logging.

        This method handles all error sending logic and gracefully skips sending
        if the connection is closed or if the error is about connection closure.
        """
        # Don't try to send errors if connection is already closed
        if not self.is_connected():
            logger.debug(
                f"⚠️ Skipping error send for stream {error.stream_id}: connection is closed"
            )
            return

        error_payload = {
            "error_type": error.error_type,
            "message": error.message,
            "stream_id": error.stream_id,
            "recoverable": error.recoverable,
            "timestamp": error.timestamp,
        }

        try:
            error_message = Message(type=MessageType.ERROR, body=error_payload)
            await self.send_json_safe(error_message.model_dump(), is_error=True)
            logger.error(f"🛑 Sent error to client stream {error.stream_id}: {error_payload}")

        # Catch any errors (e.g., Unexpected WebSocket issues, Message creation, JSON serialization)
        # Log but don't propagate - error sending should be best-effort
        except WebSocketError as e:
            logger.debug(
                f"⚠️ Could not send error to WebSocket for stream {error.stream_id}: {e.message}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to send error to WebSocket for stream {error.stream_id}: {e}")
            traceback.print_exc()
