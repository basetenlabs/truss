#!/usr/bin/env python3
"""Local invocation script for the deployed voice_agent_mocked chain.

Usage:

    export BASETEN_API_KEY=...
    export CHAIN_URL="wss://chain-<id>.api.baseten.co/development/websocket"
    python client.py

Sends a 1024-byte buffer to the entrypoint and prints what comes back. Expected:
the round-trip ``Whisper → LLM → TTS`` produces ``b"echo: text-1024"``.
"""

import asyncio
import os
import sys

import websockets


async def main() -> None:
    url = os.environ.get("CHAIN_URL")
    api_key = os.environ.get("BASETEN_API_KEY")
    if not url or not api_key:
        print("Error: CHAIN_URL and BASETEN_API_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Api-Key {api_key}"}
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(b"a" * 1024)
        reply = await ws.recv()
        print(f"reply ({type(reply).__name__}, {len(reply)} bytes):")
        print(repr(reply))
        expected = b"echo: text-1024"  # STT-LLM-TTS mock pipeline output
        if reply == expected:
            print(f"\n✓ Match: {expected!r}")
        else:
            print(f"\n✗ Mismatch — expected {expected!r}")


if __name__ == "__main__":
    asyncio.run(main())
