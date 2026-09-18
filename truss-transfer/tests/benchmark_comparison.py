#!/usr/bin/env python3
"""
Compare Rust truss_transfer vs ffmpeg-python for audio processing.
"""

import asyncio
import tempfile
import time

import ffmpeg
import numpy as np
import requests
import truss_transfer

processor = truss_transfer.MultimodalProcessor()
session = requests.Session()


async def process_with_truss_transfer(audio_url, sample_rate=16000, channels=1):
    """Process audio using Rust truss_transfer including download."""
    audio_config = (
        truss_transfer.AudioConfig()
        .with_sample_rate(sample_rate)
        .with_channels(channels)
    )

    start = time.perf_counter()
    result, timing = await processor.process_audio_from_url(audio_url, audio_config)
    end = time.perf_counter()

    return result, timing, (end - start) * 1_000_000


def process_with_ffmpeg_python(audio_url, sample_rate=16000, channels=1):
    """Process audio using ffmpeg-python including download."""
    start = time.perf_counter()

    # Download audio
    response = session.get(audio_url)
    audio_bytes = response.content

    # Process audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()

        # Use ffmpeg-python to process audio
        try:
            stream = ffmpeg.input(temp_file.name)
            stream = ffmpeg.output(
                stream,
                "pipe:",
                format="f32le",
                acodec="pcm_f32le",
                ac=channels,
                ar=sample_rate,
            )
            out, err = stream.run(capture_stdout=True, capture_stderr=True)

            if not out:
                raise ValueError("ffmpeg-python produced no output")

            # Convert to numpy array
            audio_np = np.frombuffer(out, dtype=np.float32)

            end = time.perf_counter()

            return audio_np, (end - start) * 1_000_000
        except ffmpeg.Error as e:
            raise ValueError(f"ffmpeg-python failed: {e}")


async def benchmark_comparison(audio_url, sample_rate=16000, channels=1, iterations=5):
    """Compare truss_transfer vs ffmpeg-python."""

    print(f"\n{'=' * 70}")
    print(f"Benchmark Comparison: {audio_url}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Iterations: {iterations}")
    print(f"{'=' * 70}")

    # Benchmark truss_transfer
    print("\n" + "=" * 70)
    print("Testing truss_transfer (Rust)")
    print("=" * 70)

    truss_times = []
    truss_samples = []

    for i in range(iterations):
        result, timing, elapsed = await process_with_truss_transfer(
            audio_url, sample_rate, channels
        )

        truss_times.append(elapsed)
        truss_samples.append(len(result))

        print(f"  Run {i + 1}/{iterations}: {elapsed:.0f} µs, {len(result):,} samples")

    truss_avg = np.mean(truss_times)
    truss_std = np.std(truss_times)
    truss_min = np.min(truss_times)
    truss_max = np.max(truss_times)
    truss_median = np.median(truss_times)

    print("\ntruss_transfer Results:")
    print(f"  Average: {truss_avg:.0f} µs")
    print(f"  Std dev: {truss_std:.0f} µs")
    print(f"  Median:  {truss_median:.0f} µs")
    print(f"  Min:     {truss_min:.0f} µs")
    print(f"  Max:     {truss_max:.0f} µs")
    print(f"  Samples: {truss_samples[0]:,}")

    # Benchmark ffmpeg-python
    print("\n" + "=" * 70)
    print("Testing ffmpeg-python (Python)")
    print("=" * 70)

    ffmpeg_times = []
    ffmpeg_samples = []

    for i in range(iterations):
        result, elapsed = process_with_ffmpeg_python(audio_url, sample_rate, channels)

        ffmpeg_times.append(elapsed)
        ffmpeg_samples.append(len(result))

        print(f"  Run {i + 1}/{iterations}: {elapsed:.0f} µs, {len(result):,} samples")

    ffmpeg_avg = np.mean(ffmpeg_times)
    ffmpeg_std = np.std(ffmpeg_times)
    ffmpeg_min = np.min(ffmpeg_times)
    ffmpeg_max = np.max(ffmpeg_times)
    ffmpeg_median = np.median(ffmpeg_times)

    print("\nffmpeg-python Results:")
    print(f"  Average: {ffmpeg_avg:.0f} µs")
    print(f"  Std dev: {ffmpeg_std:.0f} µs")
    print(f"  Median:  {ffmpeg_median:.0f} µs")
    print(f"  Min:     {ffmpeg_min:.0f} µs")
    print(f"  Max:     {ffmpeg_max:.0f} µs")
    print(f"  Samples: {ffmpeg_samples[0]:,}")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    speedup = ffmpeg_avg / truss_avg
    improvement = ((ffmpeg_avg - truss_avg) / ffmpeg_avg) * 100

    print(f"  truss_transfer: {truss_avg:.0f} µs (±{truss_std:.0f})")
    print(f"  ffmpeg-python: {ffmpeg_avg:.0f} µs (±{ffmpeg_std:.0f})")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"{'=' * 70}")

    # Verify both produce identical results
    samples_match = np.allclose(truss_samples[0], ffmpeg_samples[0], rtol=1e-5)
    max_diff = np.max(np.abs(truss_samples[0] - ffmpeg_samples[0]))
    print("  Verification:")
    print(f"    Samples match: {samples_match}")
    print(f"    Max difference: {max_diff:.10f}")

    return {
        "truss_avg": truss_avg,
        "truss_std": truss_std,
        "ffmpeg_avg": ffmpeg_avg,
        "ffmpeg_std": ffmpeg_std,
        "speedup": speedup,
        "improvement": improvement,
        "samples_match": samples_match,
        "max_diff": max_diff,
    }


async def main():
    """Run comparison benchmarks."""

    # Test URLs
    TEST_URLS = {
        "small_mp3": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",  # ~1.5 MB
        "medium_m4a": "https://test-audios-public.s3.us-west-2.amazonaws.com/30-sec-01-podcast.m4a",  # ~3 MB
        "m4a": "https://storage.googleapis.com/public-lyrebird-test/v4_concat_1hr_16k_mono.wav",
    }

    results = {}

    # warmup
    await benchmark_comparison(
        TEST_URLS["small_mp3"], sample_rate=16000, channels=1, iterations=1
    )

    for test_name, url in TEST_URLS.items():
        print(f"\n{'#' * 70}")
        print(f"# Testing: {test_name}")
        print(f"{'#' * 70}")

        metrics = await benchmark_comparison(
            url, sample_rate=16000, channels=1, iterations=5
        )
        results[test_name] = metrics

    # Summary
    print(f"\n{'#' * 70}")
    print("# SUMMARY")
    print(f"{'#' * 70}")
    for test_name, metrics in results.items():
        print(f"\n{test_name}:")
        print(f"  truss_transfer: {metrics['truss_avg']:.0f} µs")
        print(f"  ffmpeg-python: {metrics['ffmpeg_avg']:.0f} µs")
        print(
            f"  Speedup:  {metrics['speedup']:.2f}x ({metrics['improvement']:.1f}% {'faster' if metrics['improvement'] > 0 else 'slower'})"
        )
        print(f"  Verification: {metrics['samples_match']}")
        print(f"  Max difference: {metrics['max_diff']:.10f}")


if __name__ == "__main__":
    asyncio.run(main())
