import json
import os
import time

import pytest
import requests
from baseten_performance_client import PerformanceClient

api_key = os.environ.get("BASETEN_API_KEY")
base_url_stream = "https://model-jwdrdrm3.api.baseten.co/environments/production/sync"  # /v1/chat/completions
payload = {
    "model": "my_model",
    "messages": [{"role": "user", "content": "Tell me a very very long story."}],
    "stream": True,
    "max_tokens": 100,
    "stream_options": {"include_usage": True},
    "temperature": 0.0,
}


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
    client = PerformanceClient(base_url_stream, api_key=api_key)

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
    print(f"Received {len(events)} events in {times[-1]:.2f} seconds")
    return times


@pytest.mark.skipif(
    not is_deployment_reachable(base_url_stream), reason="Deployment not reachable"
)
async def test_streaming_async():
    client = PerformanceClient(base_url_stream, api_key=api_key)

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
    print(f"Received {len(events)} events in {times[-1]:.2f} seconds")
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
    response = requests.post(
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
    print(f"Received {len(events)} events in {times[-1]:.2f} seconds")
    return times


if __name__ == "__main__":
    t1 = test_streaming()
    import asyncio

    t2 = asyncio.run(test_streaming_async())
    t3 = test_streaming_requests_python()
    print(f"Sync: {t1}, Async: {t2}, Requests: {t3}")
