#!/usr/bin/env python3
"""
Benchmark script to compare Streaming ASR Truss vs Chain.

Usage:
    # Quick test against Truss only
    python benchmark_asr.py --truss-url wss://model-5qeok6r3.api.baseten.co/environments/production/websocket

    # Compare Truss vs Chain
    python benchmark_asr.py \
        --truss-url wss://model-5qeok6r3.api.baseten.co/environments/production/websocket \
        --chain-url wss://chain-XXXXX.api.baseten.co/environments/production/websocket

    # With custom audio file
    python benchmark_asr.py --truss-url ... --audio-file path/to/audio.wav

    # Verbose mode with live transcription output
    python benchmark_asr.py --truss-url ... --verbose
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import httpx
import numpy as np
import soundfile as sf
import websockets

# Audio config
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Must be exactly 512 for Silero VAD at 16kHz
CHUNK_DURATION = CHUNK_SIZE / SAMPLE_RATE  # ~0.032 seconds

WEB_SOCKET_OPEN_TIMEOUT = 120.0
WEB_SOCKET_TIMEOUT = 30.0
HTTP_AUDIO_DOWNLOAD_TIMEOUT = 30.0


@dataclass
class TranscriptEvent:
    """A single transcript event with timing info."""

    timestamp_ms: float  # When we received this
    is_final: bool
    text: str
    start_time: float  # Audio start time
    end_time: float  # Audio end time
    latency_ms: float  # Time from audio_send_start to receipt


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    endpoint_name: str
    endpoint_url: str
    audio_file: str
    audio_duration_s: float = 0.0
    success: bool = False
    error: Optional[str] = None

    # Timing metrics
    connection_time_ms: float = 0.0
    first_partial_latency_ms: float = 0.0
    first_final_latency_ms: float = 0.0
    last_final_latency_ms: float = 0.0
    total_time_ms: float = 0.0

    # Message counts
    partial_count: int = 0
    final_count: int = 0
    total_messages: int = 0

    # Data transfer
    bytes_sent: int = 0
    bytes_received: int = 0
    chunks_sent: int = 0

    # Latency stats
    partial_latencies_ms: List[float] = field(default_factory=list)
    final_latencies_ms: List[float] = field(default_factory=list)

    # Transcription results
    final_transcripts: List[str] = field(default_factory=list)
    transcript_events: List[TranscriptEvent] = field(default_factory=list)

    @property
    def avg_partial_latency_ms(self) -> float:
        return (
            sum(self.partial_latencies_ms) / len(self.partial_latencies_ms)
            if self.partial_latencies_ms
            else 0.0
        )

    @property
    def avg_final_latency_ms(self) -> float:
        return (
            sum(self.final_latencies_ms) / len(self.final_latencies_ms)
            if self.final_latencies_ms
            else 0.0
        )

    @property
    def real_time_factor(self) -> float:
        """How fast we processed relative to audio duration. <1 means faster than real-time."""
        if self.audio_duration_s > 0:
            return (self.total_time_ms / 1000) / self.audio_duration_s
        return 0.0

    @property
    def best_transcript(self) -> str:
        """Get the best available transcript: finals if available, otherwise last partial."""
        if self.final_transcripts:
            return " ".join(self.final_transcripts).strip()
        # Fallback to last partial (common for continuous speech without pauses)
        partial_events = [e for e in self.transcript_events if not e.is_final]
        if partial_events:
            return partial_events[-1].text.strip()
        return ""


async def load_audio_file(audio_file_path: str) -> tuple[np.ndarray, int]:
    """Load audio file from local path or URL."""
    try:
        if audio_file_path.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=HTTP_AUDIO_DOWNLOAD_TIMEOUT) as client:
                response = await client.get(audio_file_path)
                response.raise_for_status()
                audio_bytes = response.content

            parsed_url = urllib.parse.urlparse(audio_file_path)
            path = parsed_url.path
            _, ext = os.path.splitext(path)
            if not ext:
                ext = ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name

            try:
                audio_data, sample_rate = sf.read(tmp_file_path)
            finally:
                os.unlink(tmp_file_path)
        else:
            audio_data, sample_rate = sf.read(audio_file_path)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            from scipy import signal

            num_samples = int(len(audio_data) * SAMPLE_RATE / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            print(f"  Resampled audio to {SAMPLE_RATE}Hz")

        return audio_data, SAMPLE_RATE
    except Exception as e:
        print(f"  ❌ Error loading audio file: {e}")
        raise


async def send_audio(
    audio_data: np.ndarray,
    ws,
    result: BenchmarkResult,
    realtime: bool = True,
    speed_factor: float = 1.0,
    audio_file_path: str = "",
    verbose: bool = False,
):
    """Send audio chunks to WebSocket."""
    total_chunks = (len(audio_data) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i : i + CHUNK_SIZE]

        # Pad the last chunk if necessary
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

        # Convert to int16 bytes
        int16_data = (chunk * 32767).astype(np.int16).tobytes()
        await ws.send(int16_data)

        result.bytes_sent += len(int16_data)
        result.chunks_sent += 1

        # Progress indicator
        if verbose and result.chunks_sent % 100 == 0:
            progress = result.chunks_sent / total_chunks * 100
            print(
                f"  📤 Sent {result.chunks_sent}/{total_chunks} chunks ({progress:.1f}%)", end="\r"
            )

        # Simulate real-time streaming
        if realtime:
            wait_time = CHUNK_DURATION / speed_factor
            await asyncio.sleep(wait_time)

    if verbose:
        print(f"  📤 Sent {result.chunks_sent}/{total_chunks} chunks (100.0%)    ")

    # Send end-of-audio message (must be sent as text, not bytes!)
    eod_message = {"type": "end_audio", "trace_id": audio_file_path}
    eod_json = json.dumps(eod_message)
    await ws.send(eod_json)  # Send as text (string), not binary
    result.bytes_sent += len(eod_json.encode())


async def receive_messages(
    ws,
    result: BenchmarkResult,
    audio_send_start: float,
    verbose: bool = False,
):
    """Receive and process server messages."""
    try:
        while True:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=WEB_SOCKET_TIMEOUT)
                result.bytes_received += (
                    len(response) if isinstance(response, bytes) else len(response.encode())
                )
                result.total_messages += 1

                message = json.loads(response)
                msg_type = message.get("type", "transcription")

                if msg_type == "end_audio":
                    if verbose:
                        status = message.get("body", {}).get("status", "unknown")
                        print(f"  📥 End of audio received (status: {status})")
                    if message.get("body", {}).get("status") == "finished":
                        break
                    continue

                if msg_type != "transcription":
                    if verbose:
                        print(f"  📥 Received message type: {msg_type}")
                    continue

                now = time.time() * 1000
                latency = now - audio_send_start
                is_final = message.get("is_final", False)

                # Extract text from segments
                segments = message.get("segments", [])
                if segments:
                    segment = segments[0]
                    text = segment.get("text", "")
                    start_time = segment.get("start_time", 0)
                    end_time = segment.get("end_time", 0)
                else:
                    text = ""
                    start_time = 0
                    end_time = 0

                # Record event
                event = TranscriptEvent(
                    timestamp_ms=now,
                    is_final=is_final,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    latency_ms=latency,
                )
                result.transcript_events.append(event)

                if is_final:
                    result.final_count += 1
                    result.final_latencies_ms.append(latency)

                    if result.first_final_latency_ms == 0:
                        result.first_final_latency_ms = latency
                    result.last_final_latency_ms = latency

                    for seg in segments:
                        result.final_transcripts.append(seg.get("text", ""))

                    if verbose:
                        print(
                            f"  🟢 FINAL  [{start_time:.2f}s-{end_time:.2f}s] (latency: {latency:.0f}ms): {text[:80]}{'...' if len(text) > 80 else ''}"
                        )
                else:
                    result.partial_count += 1
                    result.partial_latencies_ms.append(latency)

                    if result.first_partial_latency_ms == 0:
                        result.first_partial_latency_ms = latency

                    if verbose:
                        print(
                            f"  🟡 partial [{start_time:.2f}s-{end_time:.2f}s] (latency: {latency:.0f}ms): {text[:60]}{'...' if len(text) > 60 else ''}"
                        )

            except asyncio.TimeoutError:
                print(f"  ⚠️ Timeout waiting for messages after {WEB_SOCKET_TIMEOUT}s")
                break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"  ⚠️ WebSocket closed: {e}")


def print_detailed_stats(result: BenchmarkResult):
    """Print detailed statistics for a benchmark run."""
    print(f"\n  {'─'*56}")
    print(f"  📊 DETAILED STATISTICS")
    print(f"  {'─'*56}")

    # Audio info
    print(f"\n  🎵 Audio:")
    print(f"     Duration: {result.audio_duration_s:.2f}s")
    print(f"     Chunks sent: {result.chunks_sent:,}")
    print(f"     Bytes sent: {result.bytes_sent:,} ({result.bytes_sent/1024:.1f} KB)")

    # Connection & Timing
    print(f"\n  ⏱️  Timing:")
    print(f"     Connection time: {result.connection_time_ms:.0f}ms")
    print(f"     Total time: {result.total_time_ms:.0f}ms ({result.total_time_ms/1000:.2f}s)")
    print(
        f"     Real-time factor: {result.real_time_factor:.2f}x {'(faster than real-time)' if result.real_time_factor < 1 else '(slower than real-time)' if result.real_time_factor > 1 else ''}"
    )

    # Latency breakdown
    print(f"\n  📡 Latency:")
    print(f"     First partial: {result.first_partial_latency_ms:.0f}ms")
    print(f"     First final: {result.first_final_latency_ms:.0f}ms")
    print(f"     Last final: {result.last_final_latency_ms:.0f}ms")
    if result.partial_latencies_ms:
        print(f"     Avg partial: {result.avg_partial_latency_ms:.0f}ms")
        print(f"     Min partial: {min(result.partial_latencies_ms):.0f}ms")
        print(f"     Max partial: {max(result.partial_latencies_ms):.0f}ms")
    if result.final_latencies_ms:
        print(f"     Avg final: {result.avg_final_latency_ms:.0f}ms")
        print(f"     Min final: {min(result.final_latencies_ms):.0f}ms")
        print(f"     Max final: {max(result.final_latencies_ms):.0f}ms")

    # Message counts
    print(f"\n  📨 Messages:")
    print(f"     Total received: {result.total_messages}")
    print(f"     Partial transcripts: {result.partial_count}")
    print(f"     Final transcripts: {result.final_count}")
    print(f"     Bytes received: {result.bytes_received:,} ({result.bytes_received/1024:.1f} KB)")

    # Transcription result
    print(f"\n  📝 Transcription:")
    full_text = result.best_transcript
    if full_text:
        # Word count
        word_count = len(full_text.split())
        char_count = len(full_text)
        source = "finals" if result.final_transcripts else "last partial"
        print(f"     Source: {source}")
        print(f"     Words: {word_count}, Characters: {char_count}")
        print(f"     Text: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")
    else:
        print(f"     (No transcription received)")

    print(f"  {'─'*56}")


async def run_benchmark(
    endpoint_name: str,
    endpoint_url: str,
    audio_file_path: str,
    metadata: dict,
    api_key: Optional[str] = None,
    realtime: bool = True,
    speed_factor: float = 1.0,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark against an endpoint."""
    result = BenchmarkResult(
        endpoint_name=endpoint_name,
        endpoint_url=endpoint_url,
        audio_file=audio_file_path,
    )

    headers = {}
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"

    start_time = time.time() * 1000

    try:
        print(f"\n{'='*60}")
        print(f"🚀 BENCHMARK: {endpoint_name}")
        print(f"{'='*60}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  URL: {endpoint_url}")
        print(f"  Audio: {audio_file_path}")
        print(f"  Mode: {'realtime' if realtime else 'fast'} (speed: {speed_factor}x)")

        # Load audio
        print(f"\n  📂 Loading audio...")
        audio_data, _ = await load_audio_file(audio_file_path)
        result.audio_duration_s = len(audio_data) / SAMPLE_RATE
        print(f"  ✅ Audio loaded: {result.audio_duration_s:.2f}s ({len(audio_data):,} samples)")

        # Connect
        print(f"\n  🔌 Connecting to WebSocket...")
        connect_start = time.time() * 1000

        async with websockets.connect(
            endpoint_url,
            additional_headers=headers if headers else None,
            open_timeout=WEB_SOCKET_OPEN_TIMEOUT,
        ) as ws:
            result.connection_time_ms = time.time() * 1000 - connect_start
            print(f"  ✅ Connected in {result.connection_time_ms:.0f}ms")

            # Send metadata
            metadata_json = json.dumps(metadata)
            print(f"\n  📤 Sending metadata...")
            if verbose:
                print(f"     {metadata_json}")
            await ws.send(metadata_json)
            result.bytes_sent += len(metadata_json.encode())

            # Stream audio and receive responses
            print(f"\n  🎤 Streaming audio...")
            audio_send_start = time.time() * 1000

            await asyncio.gather(
                send_audio(
                    audio_data, ws, result, realtime, speed_factor, audio_file_path, verbose
                ),
                receive_messages(ws, result, audio_send_start, verbose),
            )

            result.total_time_ms = time.time() * 1000 - start_time
            result.success = True

            # Print detailed stats
            print_detailed_stats(result)

    except Exception as e:
        result.error = str(e)
        result.total_time_ms = time.time() * 1000 - start_time
        print(f"\n  ❌ Error: {e}")
        import traceback

        traceback.print_exc()

    return result


def compare_results(truss_result: BenchmarkResult, chain_result: BenchmarkResult):
    """Compare results between Truss and Chain."""
    print(f"\n{'='*70}")
    print("📊 COMPARISON: Truss vs Chain")
    print(f"{'='*70}")

    # Latency comparison
    print("\n⏱️  Latency Comparison:")
    print(f"{'Metric':<30} {'Truss':>12} {'Chain':>12} {'Diff':>12} {'Winner':>8}")
    print("-" * 74)

    metrics = [
        ("Connection (ms)", truss_result.connection_time_ms, chain_result.connection_time_ms),
        (
            "First partial (ms)",
            truss_result.first_partial_latency_ms,
            chain_result.first_partial_latency_ms,
        ),
        (
            "First final (ms)",
            truss_result.first_final_latency_ms,
            chain_result.first_final_latency_ms,
        ),
        (
            "Avg partial (ms)",
            truss_result.avg_partial_latency_ms,
            chain_result.avg_partial_latency_ms,
        ),
        ("Avg final (ms)", truss_result.avg_final_latency_ms, chain_result.avg_final_latency_ms),
        ("Total time (ms)", truss_result.total_time_ms, chain_result.total_time_ms),
        ("Real-time factor", truss_result.real_time_factor, chain_result.real_time_factor),
    ]

    for name, truss_val, chain_val in metrics:
        diff = truss_val - chain_val
        diff_str = f"{diff:+.1f}" if diff != 0 else "0"
        if diff < 0:
            winner = "Truss"
            indicator = "🟢"
        elif diff > 0:
            winner = "Chain"
            indicator = "🔴"
        else:
            winner = "Tie"
            indicator = "⚪"
        print(
            f"{name:<30} {truss_val:>12.1f} {chain_val:>12.1f} {indicator}{diff_str:>10} {winner:>8}"
        )

    # Message count comparison
    print("\n📨 Message Count Comparison:")
    print(f"{'Metric':<30} {'Truss':>12} {'Chain':>12}")
    print("-" * 54)
    print(
        f"{'Partial transcripts':<30} {truss_result.partial_count:>12} {chain_result.partial_count:>12}"
    )
    print(
        f"{'Final transcripts':<30} {truss_result.final_count:>12} {chain_result.final_count:>12}"
    )
    print(
        f"{'Total messages':<30} {truss_result.total_messages:>12} {chain_result.total_messages:>12}"
    )

    # Transcript comparison (uses finals if available, otherwise last partial)
    print("\n📝 Transcript Comparison:")
    truss_text = truss_result.best_transcript
    chain_text = chain_result.best_transcript
    truss_source = "finals" if truss_result.final_transcripts else "last partial"
    chain_source = "finals" if chain_result.final_transcripts else "last partial"

    if truss_text == chain_text:
        print("   ✅ Transcripts MATCH exactly!")
        print(f"   Source: Truss={truss_source}, Chain={chain_source}")
        print(f"   Text: {truss_text[:150]}{'...' if len(truss_text) > 150 else ''}")
    else:
        print("   ⚠️ Transcripts DIFFER:")
        print(
            f"   Truss ({truss_source}, {len(truss_text)} chars): {truss_text[:100]}{'...' if len(truss_text) > 100 else ''}"
        )
        print(
            f"   Chain ({chain_source}, {len(chain_text)} chars): {chain_text[:100]}{'...' if len(chain_text) > 100 else ''}"
        )


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Streaming ASR endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python benchmark_asr.py --truss-url wss://model-XXX.api.baseten.co/environments/production/websocket

  # Verbose mode with live output
  python benchmark_asr.py --truss-url ... --verbose

  # Fast mode (don't wait for real-time)
  python benchmark_asr.py --truss-url ... --no-realtime

  # Compare Truss vs Chain
  python benchmark_asr.py --truss-url ... --chain-url ...
        """,
    )
    parser.add_argument("--truss-url", required=True, help="Truss WebSocket URL")
    parser.add_argument("--chain-url", help="Chain WebSocket URL (optional, for comparison)")
    parser.add_argument(
        "--audio-file",
        default="https://test-audios-public.s3.us-west-2.amazonaws.com/test_audio_en.wav",
        help="Audio file path or URL (default: test_audio_en.wav)",
    )
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY"), help="Baseten API key")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed factor (default: 1.0)"
    )
    parser.add_argument("--no-realtime", action="store_true", help="Send audio as fast as possible")
    parser.add_argument("--language", default="en", help="Audio language (default: en)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show live transcription output"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("🎙️  STREAMING ASR BENCHMARK")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not args.api_key:
        print("⚠️  Warning: No API key provided. Set BASETEN_API_KEY or use --api-key")

    # Metadata for the request
    metadata = {
        "whisper_params": {
            "audio_language": args.language,
        },
        "streaming_params": {
            "enable_partial_transcripts": True,
        },
    }

    print(f"\nConfiguration:")
    print(f"  Language: {args.language}")
    print(f"  Realtime: {not args.no_realtime}")
    print(f"  Speed: {args.speed}x")
    print(f"  Verbose: {args.verbose}")

    # Run Truss benchmark
    truss_result = await run_benchmark(
        endpoint_name="ASR Truss",
        endpoint_url=args.truss_url,
        audio_file_path=args.audio_file,
        metadata=metadata,
        api_key=args.api_key,
        realtime=not args.no_realtime,
        speed_factor=args.speed,
        verbose=args.verbose,
    )

    # Run Chain benchmark if provided
    chain_result = None
    if args.chain_url:
        chain_result = await run_benchmark(
            endpoint_name="ASR Chain",
            endpoint_url=args.chain_url,
            audio_file_path=args.audio_file,
            metadata=metadata,
            api_key=args.api_key,
            realtime=not args.no_realtime,
            speed_factor=args.speed,
            verbose=args.verbose,
        )

        # Compare results
        if truss_result.success and chain_result.success:
            compare_results(truss_result, chain_result)

    # Final Summary
    print(f"\n{'='*60}")
    print("🏁 BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if truss_result.success:
        print(f"\n✅ Truss: SUCCESS")
        print(f"   Total time: {truss_result.total_time_ms:.0f}ms")
        print(f"   Real-time factor: {truss_result.real_time_factor:.2f}x")
        print(f"   Partials: {truss_result.partial_count}, Finals: {truss_result.final_count}")
    else:
        print(f"\n❌ Truss: FAILED - {truss_result.error}")

    if chain_result:
        if chain_result.success:
            print(f"\n✅ Chain: SUCCESS")
            print(f"   Total time: {chain_result.total_time_ms:.0f}ms")
            print(f"   Real-time factor: {chain_result.real_time_factor:.2f}x")
            print(f"   Partials: {chain_result.partial_count}, Finals: {chain_result.final_count}")
        else:
            print(f"\n❌ Chain: FAILED - {chain_result.error}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
