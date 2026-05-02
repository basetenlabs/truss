"""STTMock — minimal CPU-only WS-fronted Truss that mocks streaming STT.

Protocol (mirrors Baseten Whisper STT shape, but trivial):
- Client connects WS, optionally sends a JSON metadata frame (ignored).
- Client sends audio chunks (bytes).
- For each chunk, server replies with `{"text": "text-{N}"}` where N is the
  byte length of the chunk.
"""

import json


class Model:
    def __init__(self, **kwargs) -> None:
        pass

    async def websocket(self, ws):
        try:
            while True:
                # fastapi.WebSocket.receive() returns a dict like
                # {"type": "websocket.receive", "bytes": ...} or
                # {"type": "websocket.receive", "text": ...} —
                # use the typed helpers instead.
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                if "bytes" in msg and msg["bytes"] is not None:
                    audio = msg["bytes"]
                    await ws.send_text(
                        json.dumps({"text": f"text-{len(audio)}"})
                    )
                # Text frames (e.g. JSON metadata handshake) are silently consumed.
        except Exception:
            return
