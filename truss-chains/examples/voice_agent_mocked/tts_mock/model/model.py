"""TTSMock — minimal CPU-only WS-fronted Truss that mocks streaming TTS.

Protocol (mirrors Baseten Orpheus TTS shape, but trivial):
- Client connects WS.
- Client sends a JSON config frame (ignored).
- Client sends a text frame containing the text to synthesize.
- Server replies with `<text>.encode()` as a single bytes frame, then closes.
"""

import json


class Model:
    def __init__(self, **kwargs) -> None:
        pass

    async def websocket(self, ws):
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                text = msg.get("text")
                if text is None:
                    continue
                # If the text frame is a JSON config blob, swallow and wait for
                # the actual text-to-synthesize frame.
                try:
                    json.loads(text)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
                await ws.send_bytes(text.encode())
        except Exception:
            return
