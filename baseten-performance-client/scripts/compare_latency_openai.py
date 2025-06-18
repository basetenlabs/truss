import asyncio
import csv
import os
import threading
import time

import numpy as np
import psutil
from baseten_performance_client import OpenAIEmbeddingsResponse, PerformanceClient
from openai import AsyncOpenAI

# Configuration
api_key = os.environ.get("BASETEN_API_KEY")
if not api_key:
    raise ValueError("BASETEN_API_KEY environment variable not set.")
api_base_embed = "https://model-e3m0299q.api.baseten.co/environments/production/sync"

# Benchmark settings: list of lengths to test.
benchmark_lengths = [128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608]
micro_batch_size = (
    128  # For AsyncOpenAI client; also used for the PerformanceClient batch
)

client_b = PerformanceClient(api_key=api_key, base_url=api_base_embed)
client_oai = AsyncOpenAI(api_key=api_key, base_url=api_base_embed, timeout=1024)


# --- CPU Monitor Function ---
def cpu_monitor(cpu_usage_list, stop_event, interval=0.1):
    """Collects CPU usage readings for the current process.
    The value represents the process's CPU utilization as a percentage,
    where 100% means one full core is utilized by the process.
    This can exceed 100% on multi-core systems if the process uses multiple cores.
    """
    process = psutil.Process(os.getpid())
    # num_cores = psutil.cpu_count() # Not needed for this calculation
    process.cpu_percent(
        interval=None
    )  # Initialize baseline; the first subsequent call will measure usage since this point.
    while not stop_event.is_set():
        try:
            usage = process.cpu_percent(interval=interval)
            if (
                usage is not None
            ):  # cpu_percent can return None if called too quickly with interval=None
                cpu_usage_list.append(usage)
        except psutil.NoSuchProcess:  # Process might have ended
            break
        except Exception as e:
            print(f"Error in CPU monitor: {e}")
            break


async def run_baseten_benchmark(length):
    """Run a single PerformanceClient benchmark with CPU monitoring."""
    # Prepare input data
    full_input_texts = ["Hello world"] * length

    # Warm-up run
    _ = await client_b.async_embed(
        input=full_input_texts[:2048],
        model="text-embedding-3-small",
        max_concurrent_requests=512,
        batch_size=micro_batch_size,
    )

    # Setup CPU monitor
    cpu_readings = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=cpu_monitor, args=(cpu_readings, stop_event, 0.1), daemon=True
    )
    monitor_thread.start()

    # Timed run
    time_start = time.monotonic()
    response = await client_b.async_embed(
        input=full_input_texts,
        model="text-embedding-3-small",
        max_concurrent_requests=512,
        batch_size=micro_batch_size,
    )
    embeddings_array = response.numpy()
    time_end = time.monotonic()
    duration = time_end - time_start

    # Stop CPU monitoring
    stop_event.set()
    monitor_thread.join(timeout=2)
    max_cpu = max(cpu_readings) if cpu_readings else 0.0
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0.0

    # Basic validations
    assert isinstance(response, OpenAIEmbeddingsResponse)
    assert len(response.data) == length
    assert embeddings_array.shape[0] == length

    return {
        "client": "PerformanceClient",
        "length": length,
        "duration": duration,
        "max_cpu": max_cpu,
        "avg_cpu": avg_cpu,
        "readings": len(cpu_readings),
    }


async def run_asyncopenai_benchmark(length):
    """Run a single AsyncOpenAI benchmark with CPU monitoring."""
    # Prepare input data
    input_texts = ["Hello world"] * micro_batch_size
    num_tasks = length // micro_batch_size
    # Limit concurrent requests,
    # otherwise openai client falls appart, feel free to remove and see effects.
    semaphore = asyncio.Semaphore(512)

    async def create():
        async with semaphore:
            return await client_oai.embeddings.create(
                input=input_texts, model="text-embedding-3-small"
            )

    # Warm-up
    _ = await asyncio.gather(*[create() for _ in range(min(num_tasks, 16))])

    # Setup CPU monitor
    cpu_readings = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=cpu_monitor, args=(cpu_readings, stop_event, 0.1), daemon=True
    )
    monitor_thread.start()

    # Timed run
    time_start = time.monotonic()
    api_responses = await asyncio.gather(*[create() for _ in range(num_tasks)])
    all_embeddings = []
    for res in api_responses:
        for emb in res.data:
            all_embeddings.append(emb.embedding)
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    time_end = time.monotonic()
    duration = time_end - time_start

    # Stop CPU monitoring
    stop_event.set()
    monitor_thread.join(timeout=2)
    max_cpu = max(cpu_readings) if cpu_readings else 0.0
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0.0

    assert len(all_embeddings) == length
    assert embeddings_array.shape[0] == length

    return {
        "client": "AsyncOpenAI",
        "length": length,
        "duration": duration,
        "max_cpu": max_cpu,
        "avg_cpu": avg_cpu,
        "readings": len(cpu_readings),
    }


async def run_all_benchmarks():
    """Runs benchmarks for all provided lengths for both clients and returns collected results."""
    results = []
    for length in benchmark_lengths:
        print(
            f"\nRunning benchmark for length: {length}, concurrent requests {length // micro_batch_size}"
        )
        res_baseten = await run_baseten_benchmark(length)
        print(
            f"PerformanceClient: duration={res_baseten['duration']:.4f} s, max_cpu={res_baseten['max_cpu']:.2f}%"
        )
        res_async = await run_asyncopenai_benchmark(length)
        print(
            f"AsyncOpenAI  : duration={res_async['duration']:.4f} s, max_cpu={res_async['max_cpu']:.2f}%"
        )
        results.append(res_baseten)
        results.append(res_async)
    return results


def write_results_csv(results, filename="benchmark_results.csv"):
    """Writes benchmark results to a CSV file."""
    fieldnames = ["client", "length", "duration", "max_cpu", "avg_cpu", "readings"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"\nBenchmark results saved to {filename}")


async def main():
    results = await run_all_benchmarks()
    write_results_csv(results)


if __name__ == "__main__":
    print("Starting benchmark comparison for PerformanceClient and AsyncOpenAI")
    asyncio.run(main())
