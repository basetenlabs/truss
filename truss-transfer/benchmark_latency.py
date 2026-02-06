#!/usr/bin/env python3
"""
Benchmark to compare latency between pipe-based and tempfile-based audio processing.
"""

import time

import numpy as np
import truss_transfer

# Test audio URLs (different formats and sizes)
TEST_URLS = {
    "small_mp3": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",  # ~1.5 MB
    "medium_m4a": "https://test-audios-public.s3.us-west-2.amazonaws.com/30-sec-01-podcast.m4a",  # ~3 MB
}


def benchmark_method(processor, url, method_name, use_pipes, iterations=5):
    """Benchmark a specific method."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {method_name}")
    print(f"URL: {url}")
    print(f"Use pipes: {use_pipes}")
    print(f"Iterations: {iterations}")
    print(f"{'=' * 60}")

    # Download audio once
    print("Downloading audio...")
    audio_bytes = processor.download_bytes(url)
    print(f"Downloaded {len(audio_bytes):,} bytes")

    # Create config with specified pipe setting
    audio_config = truss_transfer.AudioConfig()
    audio_config.use_pipes = use_pipes

    # Warmup run
    print("Warmup run...")
    try:
        audio_array, timing = processor.process_audio_from_bytes(
            audio_bytes, audio_config
        )
        print(f"Warmup successful - {timing}")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return None, None

    # Benchmark runs
    latencies = []
    sample_counts = []
    timing_infos = []

    for i in range(iterations):
        start = time.perf_counter()
        try:
            result, timing = processor.process_audio_from_bytes(
                audio_bytes, audio_config
            )
            end = time.perf_counter()

            latency_us = (end - start) * 1_000_000
            latencies.append(latency_us)
            sample_counts.append(len(result))
            timing_infos.append(timing)

            print(
                f"  Run {i + 1}/{iterations}: {latency_us:.0f} µs, {len(result):,} samples"
            )
            print(f"    Timing: {timing}")
        except Exception as e:
            print(f"  Run {i + 1}/{iterations}: FAILED - {e}")
            return None, None

    # Calculate statistics
    if latencies:
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        median_latency = np.median(latencies)

        # Average timing info
        avg_total_us = np.mean([t.total_us for t in timing_infos])
        avg_download_us = np.mean([t.download_us for t in timing_infos])
        avg_processing_us = np.mean([t.processing_us for t in timing_infos])
        avg_format_detection_us = np.mean([t.format_detection_us for t in timing_infos])

        print(f"\nResults for {method_name}:")
        print(f"  Average: {avg_latency:.0f} µs")
        print(f"  Std dev: {std_latency:.0f} µs")
        print(f"  Median:  {median_latency:.0f} µs")
        print(f"  Min:     {min_latency:.0f} µs")
        print(f"  Max:     {max_latency:.0f} µs")
        print(f"  Samples: {sample_counts[0]:,}")
        print("\n  Average Timing Breakdown:")
        print(f"    Total:              {avg_total_us:.0f} µs")
        print(f"    Download:           {avg_download_us:.0f} µs")
        print(f"    Processing:         {avg_processing_us:.0f} µs")
        print(f"    Format detection:   {avg_format_detection_us:.0f} µs")

        return avg_latency, std_latency
    else:
        return None, None


def main():
    """Run benchmarks comparing pipes vs tempfiles."""
    processor = truss_transfer.MultimodalProcessor(timeout_secs=300)

    results = {}

    for test_name, url in TEST_URLS.items():
        print(f"\n{'#' * 60}")
        print(f"# Testing: {test_name}")
        print(f"{'#' * 60}")

        # Benchmark with pipes
        pipe_avg, pipe_std = benchmark_method(
            processor, url, f"{test_name} (pipes)", use_pipes=True, iterations=5
        )

        # Benchmark with tempfiles
        temp_avg, temp_std = benchmark_method(
            processor, url, f"{test_name} (tempfile)", use_pipes=False, iterations=5
        )

        if pipe_avg is not None and temp_avg is not None:
            speedup = temp_avg / pipe_avg
            improvement = ((temp_avg - pipe_avg) / temp_avg) * 100

            print(f"\n{'=' * 60}")
            print(f"Comparison for {test_name}:")
            print(f"{'=' * 60}")
            print(f"  Pipes:    {pipe_avg:.0f} µs (±{pipe_std:.0f})")
            print(f"  Tempfile: {temp_avg:.0f} µs (±{temp_std:.0f})")
            print(f"  Speedup:  {speedup:.2f}x")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"{'=' * 60}")

            results[test_name] = {
                "pipe_avg": pipe_avg,
                "pipe_std": pipe_std,
                "temp_avg": temp_avg,
                "temp_std": temp_std,
                "speedup": speedup,
                "improvement": improvement,
            }

    # Summary
    print(f"\n{'#' * 60}")
    print("# SUMMARY")
    print(f"{'#' * 60}")
    for test_name, metrics in results.items():
        print(f"\n{test_name}:")
        print(f"  Pipes:    {metrics['pipe_avg']:.0f} µs")
        print(f"  Tempfile: {metrics['temp_avg']:.0f} µs")
        print(
            f"  Speedup:  {metrics['speedup']:.2f}x ({metrics['improvement']:.1f}% faster)"
        )


if __name__ == "__main__":
    main()
