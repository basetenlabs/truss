"""Invoke the perf streaming front-end showcase.

Sends a 3-sentence FRAME-encoded audio buffer over WS, in two flushes,
ending with FRAME|END. Prints the assigner's speaker-attributed output
so the reader can see the end-to-end legibility.

Usage::

    CHAIN_URL=wss://chain-<id>.api.baseten.co/development/websocket \\
    BASETEN_API_KEY=<key> \\
    python caller.py
"""

import asyncio
import json
import os
import sys

import websockets

_FRAMES_FLUSH_1 = [("A", "Hi there."), ("B", "Hey, hello.")]
_FRAMES_FLUSH_2 = [("A", "How are you?")]


def _encode(frames: list[tuple[str, str]]) -> bytes:
    return b"\n".join(f"FRAME|{spk}|{sent}".encode() for spk, sent in frames) + b"\n"


async def main() -> int:
    chain_url = os.environ.get("CHAIN_URL")
    api_key = os.environ.get("BASETEN_API_KEY")
    if not chain_url or not api_key:
        print("CHAIN_URL and BASETEN_API_KEY must be set.", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Api-Key {api_key}"}

    print(f"Connecting: {chain_url}")
    expected_sentences = [s for _, s in _FRAMES_FLUSH_1 + _FRAMES_FLUSH_2]
    expected_speakers = [sp for sp, _ in _FRAMES_FLUSH_1 + _FRAMES_FLUSH_2]

    async with websockets.connect(chain_url, additional_headers=headers) as ws:
        await ws.send(_encode(_FRAMES_FLUSH_1))
        await asyncio.sleep(0.1)
        await ws.send(_encode(_FRAMES_FLUSH_2))
        await ws.send(b"FRAME|END\n")
        reply_raw = await ws.recv()

    reply = json.loads(reply_raw)
    if "error" in reply:
        print(f"✗ {reply['error']}")
        return 1

    print("\nTranscriber calls:")
    for r in reply.get("transcribe_calls", []):
        body = r.get("body") or {}
        print(f"  order={body.get('order')} text={body.get('text')!r}")

    diarize_body = (reply.get("diarize") or {}).get("body") or {}
    print(f"\nDiarizer segments: {len(diarize_body.get('segments', []))}")
    for seg in diarize_body.get("segments", []):
        print(
            f"  {seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}s]: {seg['text_hint']!r}"
        )

    assigner_body = (reply.get("assigner") or {}).get("body") or {}
    items = assigner_body.get("items") or []
    print(f"\nAssigned transcript ({len(items)} items):")
    for it in items:
        print(f"  {it['speaker']}: {it['text']}")

    got_sentences = [it["text"].split(" [#")[0] for it in items]
    got_speakers = [it["speaker"] for it in items]
    ok_sent = got_sentences == expected_sentences
    ok_spk = got_speakers == expected_speakers
    print(f"\n{'✓' if ok_sent else '✗'} sentences match expected order")
    print(f"{'✓' if ok_spk else '✗'} speakers match expected order")
    return 0 if (ok_sent and ok_spk) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
