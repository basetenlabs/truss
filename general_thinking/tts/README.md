---
library_id: qwen-3-tts-streaming-base-12hz-1.7b
display_name: Qwen3 TTS 12Hz Base Streaming 1.7B
---
# Qwen3 TTS 12Hz Base Streaming 1.7B

## Example Usage
First, send an initial request with a voice name for referring to the clone. Audio should be around 10-20s in duration.
`python call.py voices add --name alex --ref-audio "https://example.com/reference.wav" --ref-text "Transcript of the reference audio.`

Once, this initial request has been completed the clone will be stored under the specified voice name and reference audio/text no longer need to be passed.
`python call.py --text "Hello world." --voice alex --stream-audio`

```python
#!/usr/bin/env python3
"""WebSocket client for Qwen3-TTS Base deployment on Baseten.

All operations go over a single WebSocket connection:
  1. Streaming TTS        (session.config → input.text → input.done → audio)
  2. Voice management     (voice.list / voice.add / voice.remove)

Set BASETEN_API_KEY in your environment before running.

Examples:
    # Streaming TTS with a built-in voice
    python call.py --text "Hello! How are you today?"

    # With a specific voice
    python call.py --text "Hello world." --voice my_voice

    # Stream PCM for lower latency
    python call.py --text "Hello world." --voice my_voice --stream-audio

    # Simulate real-time STT drip-feed
    python call.py --text "Pack my box with five dozen liquor jugs." \
        --simulate-stt --stt-delay 0.08

    # Upload a voice (local file)
    python call.py voices add --name my_voice \
        --ref-audio ./reference.wav \
        --ref-text "Transcript of the reference audio."

    # Upload a voice (from URL)
    python call.py voices add --name my_voice \
        --ref-audio "https://example.com/reference.wav" \
        --ref-text "Transcript of the reference audio."

    # List voices
    python call.py voices list

    # Remove a voice
    python call.py voices remove --name my_voice

Requirements:
    pip install websockets soundfile numpy requests
"""

import argparse
import asyncio
import base64
import io
import json
import os
import struct
import sys
import time

import numpy as np
import requests

try:
    import websockets
except ImportError:
    websockets = None

try:
    import soundfile as sf
except ImportError:
    sf = None

WS_URL = "wss://model-wx412j6q.api.baseten.co/deployment/wgl225g/websocket"
SAMPLE_RATE = 24000


def _resolve_text(value: str | None) -> str | None:
    """If *value* is a path to an existing .txt file, return its contents."""
    if value and value.endswith(".txt") and os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return f.read().strip()
    return value


def _api_key() -> str:
    key = os.getenv("BASETEN_API_KEY")
    if not key:
        sys.exit("Error: BASETEN_API_KEY environment variable is not set")
    return key


def _auth_headers() -> dict:
    return {"Authorization": f"Api-Key {_api_key()}"}


async def _ws_connect():
    if websockets is None:
        sys.exit("Missing dependency: pip install websockets")
    return await websockets.connect(
        WS_URL,
        max_size=16 * 1024 * 1024,
        additional_headers=_auth_headers(),
        open_timeout=30,
    )


# ── Voice management (over WebSocket) ───────────────────────────────────────

async def ws_voice_list() -> None:
    ws = await _ws_connect()
    print(f"[ws] Connected to {WS_URL} ...")
    try:
        await ws.send(json.dumps({"type": "voice.list"}))
        resp = json.loads(await ws.recv())

        if resp.get("type") == "error":
            print(f"[error] {resp['message']}")
            return

        builtin = resp.get("voices", [])
        uploaded = resp.get("uploaded_voices", [])

        if builtin:
            print("Built-in voices:")
            for name in builtin:
                print(f"  {name}")
        if uploaded:
            print(f"\nUploaded voices:")
            print(f"  {'Name':<20} {'Source':<10} {'Ref Text'}")
            print(f"  {'─' * 60}")
            for v in uploaded:
                print(f"  {v.get('name', '?'):<20} "
                      f"{v.get('embedding_source', ''):<10} "
                      f"{v.get('ref_text', '')[:40]}")
        if not builtin and not uploaded:
            print("No voices found.")
    finally:
        await ws.close()


async def ws_voice_add(
    name: str,
    ref_audio: str,
    ref_text: str | None = None,
    consent: str = "user_consent",
) -> None:
    ref_text = _resolve_text(ref_text)
    if ref_audio.startswith(("http://", "https://")):
        print(f"[voices] Downloading {ref_audio}...")
        r = requests.get(ref_audio, timeout=60)
        r.raise_for_status()
        audio_bytes = r.content
    else:
        with open(ref_audio, "rb") as f:
            audio_bytes = f.read()

    msg: dict = {
        "type": "voice.add",
        "name": name,
        "consent": consent,
        "audio_data": base64.b64encode(audio_bytes).decode(),
        "audio_format": "wav",
    }
    if ref_text:
        msg["ref_text"] = ref_text

    print(f"[voices] Uploading '{name}' ({len(audio_bytes):,} bytes)...")
    ws = await _ws_connect()
    try:
        await ws.send(json.dumps(msg))
        resp = json.loads(await ws.recv())

        if resp.get("type") == "error":
            print(f"[error] {resp['message']}")
        elif resp.get("success"):
            print(f"[voices] Created: {json.dumps(resp.get('voice', {}), indent=2)}")
        else:
            print(f"[error] {resp.get('error', 'unknown error')}")
    finally:
        await ws.close()


async def ws_voice_remove(name: str) -> None:
    ws = await _ws_connect()
    try:
        await ws.send(json.dumps({"type": "voice.remove", "name": name}))
        resp = json.loads(await ws.recv())

        if resp.get("type") == "error":
            print(f"[error] {resp['message']}")
        elif resp.get("success"):
            print(f"[voices] Removed '{name}'")
        else:
            print(f"[error] {resp.get('error', f'Voice {name!r} not found')}")
    finally:
        await ws.close()


# ── WebSocket streaming TTS ─────────────────────────────────────────────────

async def ws_stream(
    text: str,
    config: dict,
    output: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
) -> None:
    all_pcm: list[np.ndarray] = []
    sentence_count = 0
    first_audio_time = None
    t0 = time.perf_counter()

    is_pcm = config.get("response_format", "wav") == "pcm" or config.get("stream_audio", False)

    print(f"[ws] Connecting to {WS_URL} ...")

    try:
        ws = await _ws_connect()
    except websockets.exceptions.InvalidStatus as e:
        print(f"[error] WebSocket handshake failed: HTTP {e.response.status_code}")
        for name, value in e.response.headers.raw_items():
            print(f"        {name}: {value}")
        body = getattr(e.response, "body", None)
        if body:
            print(f"        body: {body.decode(errors='replace')[:500]}")
        raise
    except Exception as e:
        print(f"[error] WebSocket connection failed: {type(e).__name__}: {e}")
        raise

    print(f"[ws] Connected (protocol={ws.protocol})")

    try:
        config_msg = {"type": "session.config", **config}
        await ws.send(json.dumps(config_msg))
        print(f"[ws] task_type=Base  "
              f"format={config.get('response_format', 'wav')}  "
              f"stream_audio={config.get('stream_audio', False)}  "
              f"split={config.get('split_granularity', 'sentence')}")

        async def send_text():
            if simulate_stt:
                words = text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                    await asyncio.sleep(stt_delay)
                print(f"[input] Sent {len(words)} words (simulated STT, {stt_delay}s delay)")
            else:
                await ws.send(json.dumps({"type": "input.text", "text": text}))
                print(f"[input] Sent {len(text)} chars")
            await ws.send(json.dumps({"type": "input.done"}))

        sender = asyncio.create_task(send_text())

        total_bytes = 0
        current_wav_chunks: list[bytes] = []

        try:
            while True:
                msg = await ws.recv()

                if isinstance(msg, bytes):
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                    total_bytes += len(msg)
                    if is_pcm:
                        usable = len(msg) - (len(msg) % 2)
                        if usable > 0:
                            all_pcm.append(np.frombuffer(msg[:usable], dtype=np.int16))
                    else:
                        current_wav_chunks.append(msg)
                    continue

                data = json.loads(msg)
                mtype = data.get("type")

                if mtype == "audio.start":
                    current_wav_chunks = []
                    sentence_count += 1

                elif mtype == "audio.done":
                    if data.get("error", False):
                        print(f"[error] Generation failed for sentence {data['sentence_index']}")
                    elif not is_pcm and current_wav_chunks:
                        raw = b"".join(current_wav_chunks)
                        try:
                            pcm_arr, _ = sf.read(io.BytesIO(raw))
                            if pcm_arr.ndim > 1:
                                pcm_arr = pcm_arr[:, 0]
                            all_pcm.append((np.clip(pcm_arr, -1, 1) * 32767).astype(np.int16))
                        except Exception as e:
                            print(f"[error] Failed to decode sentence audio: {e}")
                    ts_info = data.get("timestamp_info")
                    if ts_info:
                        wa = ts_info.get("word_alignment", {})
                        words = wa.get("words", [])
                        starts = wa.get("word_start_times_seconds", [])
                        ends = wa.get("word_end_times_seconds", [])
                        for w, s, e in zip(words, starts, ends):
                            print(f"  [{s:.3f}–{e:.3f}] {w}")
                    current_wav_chunks = []

                elif mtype == "audio.timestamps":
                    wa = data.get("word_alignment", {})
                    words = wa.get("words", [])
                    starts = wa.get("word_start_times_seconds", [])
                    ends = wa.get("word_end_times_seconds", [])
                    for w, s, e in zip(words, starts, ends):
                        print(f"  [{s:.3f}–{e:.3f}] {w}")

                elif mtype == "session.done":
                    break

                elif mtype == "error":
                    print(f"[error] {data['message']}")

        finally:
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
    finally:
        await ws.close()

    elapsed = time.perf_counter() - t0
    ttfa = (first_audio_time - t0) if first_audio_time else None
    _save_audio(all_pcm, output, sentence_count, total_bytes, elapsed, ttfa)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_audio(
    all_pcm: list,
    output: str,
    sentence_count: int,
    total_bytes: int,
    elapsed: float,
    ttfa: float | None,
) -> None:
    if not all_pcm:
        print("\nNo audio received.")
        return

    combined = np.concatenate(all_pcm)
    audio_duration = len(combined) / SAMPLE_RATE

    if not output.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        output += ".wav"

    if sf is not None:
        sf.write(output, combined.astype(np.float32) / 32767.0, SAMPLE_RATE)
    else:
        _write_wav(output, combined)

    print(f"\n{'─' * 50}")
    print(f"  Output:     {output}")
    print(f"  Size:       {total_bytes:,} bytes")
    print(f"  Duration:   {audio_duration:.2f}s")
    print(f"  Wall time:  {elapsed:.2f}s")
    if ttfa is not None:
        print(f"  TTFA:       {ttfa * 1000:.0f}ms")
    if audio_duration > 0:
        rtf = elapsed / audio_duration
        print(f"  RTF:        {rtf:.3f}x  ({1/rtf:.1f}x realtime)")
    print(f"{'─' * 50}")


def _write_wav(path: str, pcm: np.ndarray) -> None:
    data = pcm.astype(np.int16).tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE,
                            SAMPLE_RATE * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="WebSocket client for Qwen3-TTS Base on Baseten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode")

    # ── TTS args (top-level) ─────────────────────────────────────
    p.add_argument("--text", help="Text to synthesize")
    p.add_argument("--output", "-o", default="output.wav")
    p.add_argument("--voice", default=None, help="Speaker name")
    p.add_argument("--response-format", default="wav",
                   choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--stream-audio", action="store_true",
                   help="Progressive PCM streaming (lower latency)")
    p.add_argument("--split-granularity", default="sentence",
                   choices=["sentence", "clause"])
    p.add_argument("--language", default=None)
    p.add_argument("--ref-audio", default=None,
                   help="Reference audio URL for voice cloning")
    p.add_argument("--ref-text", default=None,
                   help="Reference audio transcript (or path to a .txt file)")
    p.add_argument("--initial-codec-chunk-frames", type=int, default=None,
                   help="Initial chunk size override (larger = better quality, higher TTFA)")
    p.add_argument("--x-vector-only", action="store_true",
                   help="Use speaker embedding only, skip in-context learning")
    p.add_argument("--timestamps", default=None,
                   choices=["sync", "async"],
                   help="Enable word-level timestamps (sync: in audio.done, async: separate messages)")
    p.add_argument("--simulate-stt", action="store_true")
    p.add_argument("--stt-delay", type=float, default=0.1)

    # ── voices ───────────────────────────────────────────────────
    v_p = sub.add_parser("voices", help="Voice management (over WebSocket)")
    v_sub = v_p.add_subparsers(dest="action", required=True)

    v_sub.add_parser("list", help="List voices")

    va = v_sub.add_parser("add", help="Upload a voice (base64 audio)")
    va.add_argument("--name", required=True)
    va.add_argument("--ref-audio", required=True,
                    help="Local WAV file or URL to reference audio")
    va.add_argument("--ref-text", default=None,
                    help="Transcript of the audio, or path to a .txt file")
    va.add_argument("--consent", default="user_consent")

    vr = v_sub.add_parser("remove", help="Remove a voice")
    vr.add_argument("--name", required=True)

    args = p.parse_args()

    if args.mode == "voices":
        if args.action == "list":
            asyncio.run(ws_voice_list())
        elif args.action == "add":
            asyncio.run(ws_voice_add(
                args.name,
                args.ref_audio,
                ref_text=args.ref_text,
                consent=args.consent,
            ))
        elif args.action == "remove":
            asyncio.run(ws_voice_remove(args.name))
    else:
        if not args.text:
            p.error("--text is required for TTS")

        config: dict = {
            "task_type": "Base",
            "response_format": args.response_format,
            "speed": args.speed,
            "split_granularity": args.split_granularity,
        }
        if args.stream_audio:
            config["stream_audio"] = True
            config["response_format"] = "pcm"
        if args.initial_codec_chunk_frames is not None:
            config["initial_codec_chunk_frames"] = args.initial_codec_chunk_frames
        if args.x_vector_only:
            config["x_vector_only_mode"] = True
        if args.timestamps:
            config["timestamp_type"] = "word"
            config["timestamp_transport_strategy"] = args.timestamps
        if args.voice:
            config["voice"] = args.voice
        if args.language:
            config["language"] = args.language
        if args.ref_audio:
            config["ref_audio"] = args.ref_audio
        ref_text = _resolve_text(args.ref_text)
        if ref_text:
            config["ref_text"] = ref_text

        asyncio.run(ws_stream(
            text=args.text,
            config=config,
            output=args.output,
            simulate_stt=args.simulate_stt,
            stt_delay=args.stt_delay,
        ))


if __name__ == "__main__":
    main()
```