"""Invoke the Voice Agent Mocked showcase.

Connects to the chain over WS, sends 3 prompts (each as a MOCKAUDIO
envelope), receives back synthesized audio bytes for each, and prints
the round-trip at each stage so the legible end-to-end is visible.

Usage::

    CHAIN_URL=wss://chain-<id>.api.baseten.co/development/websocket \\
    BASETEN_API_KEY=<key> \\
    python caller.py
"""

import asyncio
import base64
import json
import os
import sys

import websockets

_HEAD = b"MOCKAUDIO|"
_TAIL = b"|END"


def _encode_envelope(text: str) -> bytes:
    return _HEAD + text.encode("utf-8") + _TAIL


def _decode_envelope(buf: bytes) -> str:
    if not (buf.startswith(_HEAD) and buf.endswith(_TAIL)):
        raise ValueError(f"not a MOCKAUDIO envelope: {buf[:32]!r}...")
    return buf[len(_HEAD) : -len(_TAIL)].decode("utf-8")


def _strip_hash(buf: bytes) -> bytes:
    return buf.rsplit(b"|H:", 1)[0]


_PROMPTS = ["What's the weather?", "Tell me a joke.", "Who are you?"]


async def _exchange(ws, prompt: str) -> dict:
    await ws.send(_encode_envelope(prompt))
    reply_raw = await ws.recv()
    reply = json.loads(reply_raw)
    if "error" in reply:
        return {"prompt": prompt, "error": reply["error"]}
    audio = base64.b64decode(reply["audio_b64"])
    decoded = _decode_envelope(_strip_hash(audio))
    return {
        "prompt": prompt,
        "user_text": reply["user_text"],
        "assistant_text": reply["assistant_text"],
        "decoded_audio_text": decoded,
        "llm_vllm_raw": reply["stages"]["llm"]["vllm_raw"],
        "llm_vllm_tokens": reply["stages"]["llm"]["vllm_token_count"],
    }


async def main() -> int:
    chain_url = os.environ.get("CHAIN_URL")
    api_key = os.environ.get("BASETEN_API_KEY")
    if not chain_url or not api_key:
        print("CHAIN_URL and BASETEN_API_KEY must be set.", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Api-Key {api_key}"}

    print(f"Connecting: {chain_url}")
    all_ok = True
    async with websockets.connect(chain_url, additional_headers=headers) as ws:
        for prompt in _PROMPTS:
            result = await _exchange(ws, prompt)
            print(f"\n>>> {prompt!r}")
            if "error" in result:
                print(f"  ✗ {result['error']}")
                all_ok = False
                continue
            print(f"  STT decoded:           {result['user_text']!r}")
            print(
                f"  LLM vLLM real call:    {result['llm_vllm_tokens']} tokens "
                f"(random weights → garbage)"
            )
            print(f"  LLM vLLM raw preview:  {result['llm_vllm_raw'][:60]!r}")
            print(f"  LLM legible response:  {result['assistant_text']!r}")
            print(f"  TTS audio decoded:     {result['decoded_audio_text']!r}")
            round_trip_ok = result["decoded_audio_text"] == result["assistant_text"]
            print(f"  {'✓' if round_trip_ok else '✗'} round-trip text matches")
            all_ok = all_ok and round_trip_ok
        await ws.send("DONE")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
