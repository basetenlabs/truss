"""leightweight benchmarking script for user-style continuous latency testing"""

import asyncio
import time

from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    base_url="https://model-e3mxl42q.api.baseten.co/environments/production/sync"
)


async def benchmark_every(
    interval=0.01,
    lb_split=128,
    tokens_per_request=500,
    sentences_per_request=1,
    n_requests=1000,
    n_users=1,
):
    async def kick_off_task():
        """kicks of a single task to measure latency."""
        try:
            t = time.time()
            await client.async_classify(
                inputs=["Hello " * tokens_per_request] * sentences_per_request,
                max_concurrent_requests=lb_split,
                batch_size=1,
                truncate=False,
            )
            # TODO: use total time or
            return [(time.time() - t)]
        except Exception as e:
            print(f"Error in task: {e}")

    async def simulate_single_user(launches_blocking=False):
        """user may launch tasks with concurrency=1 (blocking=True)
        or without any feedback at x inferval. (blocking=False)"""
        all_tasks = []
        for _ in range(n_requests):
            task = asyncio.create_task(kick_off_task())
            if launches_blocking:
                await task
            all_tasks.append(task)
            await asyncio.sleep(interval)  # Wait for 1 second before the next task

        all_times = []
        for task in all_tasks:
            try:
                if not launches_blocking:
                    result = await task
                else:
                    result = task
                all_times.extend(result)
            except Exception as e:
                print(f"Error in task completion: {e}")

        return all_times

    user_times = await asyncio.gather(
        *[simulate_single_user(launches_blocking=False) for _ in range(n_users)]
    )
    print(f"finished load test forn_users: {len(user_times)}")
    all_times = [time for sublist in user_times for time in sublist]
    print(
        f"All tasks completed. Total times collected: {len(all_times)}, Sample: {all_times[:5]}"
    )
    print(
        f"Average time per request: {sum(all_times) / len(all_times) if all_times else 0:.4f} seconds"
    )
    print(
        f"Median time per request: {sorted(all_times)[len(all_times) // 2] if all_times else 0:.4f} seconds"
    )


if __name__ == "__main__":
    asyncio.run(benchmark_every())
