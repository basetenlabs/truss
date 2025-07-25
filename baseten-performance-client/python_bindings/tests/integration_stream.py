# mypy: ignore-errors
import json
import os
import time

import numpy as np
import pytest
import requests
from baseten_performance_client import PerformanceClient

api_key = os.environ.get("BASETEN_API_KEY")
base_url_stream = "https://model-jwdrdrm3.api.baseten.co/environments/production/sync"  # /v1/chat/completions
payload = {
    "model": "my_model",
    "messages": [{"role": "user", "content": "Tell me a pretty long story."}],
    "stream": True,
    "max_tokens": 100,
    "stream_options": {"include_usage": True},
    "temperature": 0.0,
}
client = PerformanceClient(base_url_stream, api_key=api_key)
session = requests.Session()


def is_deployment_reachable(base_url, route="/v1/chat/completions", timeout=10):
    try:
        response = requests.post(
            f"{base_url}{route}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(
    not is_deployment_reachable(base_url_stream), reason="Deployment not reachable"
)
def test_streaming():
    # Define the endpoint and payload
    endpoint = "/v1/chat/completions"
    payload = {
        "model": "my_model",
        "messages": [{"role": "user", "content": "Tell me a very very long story."}],
        "stream": True,
        "max_tokens": 100,
        "stream_options": {"include_usage": True},
    }

    # Start streaming
    t = time.time()
    stream = client.stream(endpoint, payload)

    # Collect events
    events = []

    for event in stream:
        if event is None:
            break

        events.append((event, time.time() - t))

    # Check if we received the expected number of events
    assert len(events) > 0, "No events received from the stream"
    times = [event[1] for event in events]
    print(f"Received {len(events)} events in {times[-1]:.2f} seconds using sync")
    return times


@pytest.mark.skipif(
    not is_deployment_reachable(base_url_stream), reason="Deployment not reachable"
)
async def test_streaming_async():
    # Define the endpoint and payload
    endpoint = "/v1/chat/completions"

    # Start streaming
    t = time.time()
    stream = await client.async_stream(endpoint, payload)

    # Collect events
    events = []

    async for event in stream:
        if event is None:
            break

        events.append((event, time.time() - t))

    # Check if we received the expected number of events
    assert len(events) > 0, "No events received from the stream"
    times = [event[1] for event in events]
    print(f"Received {len(events)} events in {times[-1]:.2f} seconds using async")
    return times


@pytest.mark.skipif(
    not is_deployment_reachable(base_url_stream), reason="Deployment not reachable"
)
def test_streaming_requests_python():
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    endpoint = "/v1/chat/completions"

    t = time.time()

    response = session.post(
        f"{base_url_stream}{endpoint}", headers=headers, json=payload, stream=True
    )
    response.raise_for_status()

    events = []

    for line in response.iter_lines():
        if not line:
            continue
        decoded_line = line.decode("utf-8")
        if decoded_line.startswith("data: "):
            data_str = decoded_line[len("data: ") :]
            if data_str.strip() == "[DONE]":
                break
            try:
                event_data = json.loads(data_str)
                events.append((event_data, time.time() - t))
            except json.JSONDecodeError:
                print(f"Could not decode json: {data_str}")

    assert len(events) > 0, "No events received from the stream"
    times = [event[1] for event in events]
    print(f"Received {len(events)} events in {times[-1]:.2f} seconds using requests")
    return times


async def non_streaming_client():
    # Define the endpoint and payload
    endpoint = "/v1/chat/completions"
    payload = {
        "model": "my_model",
        "messages": [{"role": "user", "content": "Tell me a very very long story."}],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    t = time.time()
    await client.async_batch_post(endpoint, [payload])
    times = [time.time() - t]
    print(f"Received response in {times[-1]:.2f} seconds using non-streaming client")
    return times


# threadpool a function, and see how e.g. 5 or 10 parallel requests perform
# this is useful to see how the server handles multiple requests in parallel
# and how the client handles multiple responses in parallel
def make_function_pool(number, func, *args, **kwargs):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=number) as executor:
        futures = [executor.submit(func, *args, **kwargs) for _ in range(number)]
        results = [future.result() for future in futures]
    return np.mean(results, axis=0)


def make_async_function_pool(number, func, *args, **kwargs):
    import asyncio

    async def run_in_pool():
        tasks = [func(*args, **kwargs) for _ in range(number)]
        return await asyncio.gather(*tasks)

    return np.mean(asyncio.run(run_in_pool()), axis=0)


if __name__ == "__main__":
    t1 = test_streaming()
    import asyncio

    t2 = asyncio.run(test_streaming_async())
    t3 = test_streaming_requests_python()
    t4 = asyncio.run(non_streaming_client())
    # print(f"Sync: {t1}, Async: {t2}, Requests: {t3}")

    # run each 10 times in parallel
    number = 10
    print(f"Running sync in parallel {number} times...")
    sync_results = make_function_pool(number, test_streaming)
    print(f"Sync results: {sync_results}")
    print(f"Running async in parallel {number} times...")
    async_results = make_async_function_pool(number, test_streaming_async)
    print(f"Async results: {async_results}")
    print(f"Running requests in parallel {number} times...")
    requests_results = make_function_pool(number, test_streaming_requests_python)
    print(f"Requests results: {requests_results}")
    requests_results_non_streaming = make_async_function_pool(
        number, non_streaming_client
    )
    print(f"Non-streaming results: {requests_results_non_streaming}")
    print("All tests completed.")
