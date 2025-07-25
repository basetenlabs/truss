import asyncio
import csv
import os
import threading
import time

import baseten_performance_client
import numpy as np
import psutil
from baseten_performance_client import OpenAIEmbeddingsResponse, PerformanceClient
from openai import AsyncOpenAI

# Configuration
api_key = os.environ.get("BASETEN_API_KEY")
if not api_key:
    raise ValueError("BASETEN_API_KEY environment variable not set.")
api_base_embed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync"

# Benchmark settings: list of lengths to test.
benchmark_lengths = [16, 64, 128, 256, 384, 512, 2048, 8192, 32768, 131072]
micro_batch_size = (
    16  # For AsyncOpenAI client; also used for the PerformanceClient batch
)
client_b = PerformanceClient(api_key=api_key, base_url=api_base_embed, http_version=1)
client_b_http2 = PerformanceClient(
    api_key=api_key, base_url=api_base_embed, http_version=2
)
client_oai = AsyncOpenAI(api_key=api_key, base_url=api_base_embed, timeout=1024)


# --- CPU Monitor Function ---
def resource_monitor(cpu_usage_list, ram_usage_list, stop_event, interval=0.1):
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
            # RAM Usage (Resident Set Size in MB)
            ram_usage_mb = process.memory_info().rss / (1024 * 1024)
            ram_usage_list.append(ram_usage_mb)
        except psutil.NoSuchProcess:  # Process might have ended
            break
        except Exception as e:
            print(f"Error in CPU monitor: {e}")
            break


async def run_baseten_benchmark(length, client=client_b):
    """Run a single PerformanceClient benchmark with CPU monitoring."""
    # Prepare input data
    full_input_texts = ["Hello world"] * length

    # Warm-up run
    _ = await client.async_embed(
        input=["Hello world"] * 1024,
        model="text-embedding-3-small",
        max_concurrent_requests=512,
        batch_size=1,
    )

    # Setup CPU monitor
    cpu_readings = []
    ram_readings = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=resource_monitor,
        args=(cpu_readings, ram_readings, stop_event, 0.1),
        daemon=True,
    )
    monitor_thread.start()

    # Timed run
    time_start = time.monotonic()
    response = await client.async_embed(
        input=full_input_texts,
        model="text-embedding-3-small",
        max_concurrent_requests=512,
        batch_size=micro_batch_size,
        hedge_delay=4,
        max_chars_per_request=200,
    )
    embeddings_array = response.numpy()
    time_end = time.monotonic()
    p90 = np.percentile(response.individual_request_times, 90)
    p99 = np.percentile(response.individual_request_times, 99)
    duration = time_end - time_start

    # Stop CPU monitoring
    stop_event.set()
    monitor_thread.join(timeout=2)
    max_cpu = max(cpu_readings) if cpu_readings else 0.0
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0.0
    max_ram = max(ram_readings) if ram_readings else 0.0
    # Basic validations
    assert isinstance(response, OpenAIEmbeddingsResponse)
    assert len(response.data) == length
    assert embeddings_array.shape[0] == length
    assert list(range(length)) == [item.index for item in response.data], (
        "Response indices do not match input order."
    )

    return {
        "client": "PerformanceClient HTTP",
        "length": length,
        "duration": duration,
        "max_cpu": max_cpu,
        "avg_cpu": avg_cpu,
        "readings": len(cpu_readings),
        "max_ram": max_ram,
        "p90": p90,
        "p99": p99,
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
    ram_readings = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=resource_monitor,
        args=(cpu_readings, ram_readings, stop_event, 0.1),
        daemon=True,
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
    max_ram = max(ram_readings) if ram_readings else 0.0

    assert len(all_embeddings) == length
    assert embeddings_array.shape[0] == length

    return {
        "client": "AsyncOpenAI",
        "length": length,
        "duration": duration,
        "max_cpu": max_cpu,
        "avg_cpu": avg_cpu,
        "readings": len(cpu_readings),
        "max_ram": max_ram,
    }


async def run_all_benchmarks():
    """Runs benchmarks for all provided lengths for both clients and returns collected results."""
    results = []
    for length in benchmark_lengths:
        print(
            f"\nRunning benchmark for length: {length}, concurrent requests {length // micro_batch_size}"
        )
        res_baseten = await run_baseten_benchmark(length, client_b)
        print(
            f"PerformanceClient HTTP1: duration={res_baseten['duration']:.4f} s, max_cpu={res_baseten['max_cpu']:.2f}%, max_ram={res_baseten['max_ram']:.2f} MB p90={res_baseten['p90']:.4f}, p99={res_baseten['p99']:.4f}"
        )
        results.append(res_baseten)

        res_baseten_http2 = await run_baseten_benchmark(length, client_b_http2)
        print(
            f"PerformanceClient HTTP2: duration={res_baseten_http2['duration']:.4f} s, max_cpu={res_baseten_http2['max_cpu']:.2f}%, max_ram={res_baseten_http2['max_ram']:.2f} MB p90={res_baseten_http2['p90']:.4f}, p99={res_baseten_http2['p99']:.4f}"
        )
        res_baseten_http2["client"] = "PerformanceClient HTTP2"
        results.append(res_baseten_http2)
        res_async = await run_asyncopenai_benchmark(length)
        print(
            f"AsyncOpenAI            : duration={res_async['duration']:.4f} s, max_cpu={res_async['max_cpu']:.2f}%"
        )
        results.append(res_async)
    return results


def write_results_csv(results, filename="benchmark_results.csv"):
    """Writes benchmark results to a CSV file."""
    fieldnames = [
        "client",
        "length",
        "duration",
        "max_cpu",
        "avg_cpu",
        "readings",
        "max_ram",
    ]
    results = [{k: v for k, v in r.items() if k in fieldnames} for r in results]
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
    print(
        f"Starting benchmark comparison for PerformanceClient() and AsyncOpenAI, v{baseten_performance_client.__version__}"
    )
    asyncio.run(main())
