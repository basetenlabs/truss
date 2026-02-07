#!/usr/bin/env python3
"""
Benchmark to measure concurrency benefits of async audio processing.
"""

import asyncio
import time

import numpy as np
import truss_transfer

# Test audio URLs (different formats and sizes)
TEST_URLS = {
    "small_mp3": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",  # ~1.5 MB
    "large": "https://storage.googleapis.com/public-lyrebird-test/v4_concat_1hr_16k_mono.wav" # 100Mb
}


async def download_and_process(processor, url, audio_config):
    """Download and process audio from URL."""
    return await processor.process_audio_from_url(url, audio_config)


async def benchmark_single(processor, url, audio_config, iterations=5):
    """Benchmark single download."""
    print(f"\n{'=' * 70}")
    print("SINGLE TASK BENCHMARK")
    print(f"{'=' * 70}")
    print(f"URL: {url}")
    print(f"Iterations: {iterations}")
    print(f"{'=' * 70}")

    all_times = []

    for i in range(iterations):
        start = time.perf_counter()
        await download_and_process(processor, url, audio_config)
        elapsed = time.perf_counter() - start
        all_times.append(elapsed)
        print(f"  Iteration {i + 1}/{iterations}: {elapsed:.3f}s")

    avg_time = np.mean(all_times)
    std_time = np.std(all_times)
    min_time = np.min(all_times)
    max_time = np.max(all_times)

    print(f"\nSingle Task Results:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Std dev: {std_time:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")

    return avg_time


async def benchmark_concurrent(processor, url, audio_config, num_tasks, iterations=3):
    """Benchmark concurrent downloads with given number of tasks."""
    print(f"\n{'=' * 70}")
    print(f"CONCURRENT BENCHMARK (tasks={num_tasks})")
    print(f"{'=' * 70}")
    print(f"URL: {url}")
    print(f"Tasks: {num_tasks}")
    print(f"Iterations: {iterations}")
    print(f"{'=' * 70}")

    all_times = []

    for i in range(iterations):
        start = time.perf_counter()

        # Create all tasks and run them concurrently
        tasks = [
            download_and_process(processor, url, audio_config) for _ in range(num_tasks)
        ]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start
        all_times.append(elapsed)
        print(f"  Iteration {i + 1}/{iterations}: {elapsed:.3f}s")

    avg_time = np.mean(all_times)
    std_time = np.std(all_times)
    min_time = np.min(all_times)
    max_time = np.max(all_times)

    print(f"\nConcurrent Results (tasks={num_tasks}):")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Std dev: {std_time:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")

    return avg_time


async def main():
    """Run concurrency benchmarks."""
    processor = truss_transfer.MultimodalProcessor(timeout_secs=300)
    audio_config = truss_transfer.AudioConfig()

    await benchmark_single(processor, next(iter(TEST_URLS.items()))[1], audio_config, iterations=1)

    print(f"\n{'#' * 70}")
    print("# CONCURRENCY BENCHMARK")
    print(f"{'#' * 70}")

    for test_name, url in TEST_URLS.items():
        print(f"\n{'#' * 70}")
        print(f"# Testing: {test_name}")
        print(f"{'#' * 70}")
        print(f"URL: {url}")

        # Benchmark single task
        single_time = await benchmark_single(processor, url, audio_config, iterations=3)

        # Benchmark different numbers of concurrent tasks
        num_tasks_list = [2, 4, 8, 16, 32]
        results = {}

        for num_tasks in num_tasks_list:
            concurrent_time = await benchmark_concurrent(
                processor, url, audio_config, num_tasks, iterations=2
            )
            expected_time = single_time * num_tasks
            speedup = expected_time / concurrent_time
            efficiency = (speedup / num_tasks) * 100
            results[num_tasks] = {
                "time": concurrent_time,
                "expected_time": expected_time,
                "speedup": speedup,
                "efficiency": efficiency,
            }

        # Summary
        print(f"\n{'=' * 70}")
        print(f"SUMMARY for {test_name}")
        print(f"{'=' * 70}")
        print(f"  Single task: {single_time:.3f}s")
        print(
            f"  Expected sequential for 64 tasks: {single_time * 64:.3f}s (extrapolated)"
        )
        print()

        for num_tasks in sorted(results.keys()):
            metrics = results[num_tasks]
            print(f"  {num_tasks} tasks:")
            print(f"    Time:         {metrics['time']:.3f}s")
            print(f"    Expected:     {metrics['expected_time']:.3f}s")
            print(f"    Speedup:      {metrics['speedup']:.2f}x")
            print(f"    Efficiency:   {metrics['efficiency']:.1f}%")

        # Find optimal concurrency level
        best_num_tasks = max(results.keys(), key=lambda k: results[k]["efficiency"])
        best_efficiency = results[best_num_tasks]["efficiency"]
        best_speedup = results[best_num_tasks]["speedup"]

        print(f"\n  Optimal concurrency: {best_num_tasks} tasks")
        print(f"  Best efficiency: {best_efficiency:.1f}%")
        print(f"  Best speedup: {best_speedup:.2f}x")




if __name__ == "__main__":
    asyncio.run(main())
