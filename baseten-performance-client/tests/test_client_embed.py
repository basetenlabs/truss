import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from baseten_performance_client import (
    ClassificationResponse,
    OpenAIEmbeddingsResponse,
    PerformanceClient,
    RerankResponse,
)
from requests.exceptions import HTTPError

api_key = os.environ.get("BASETEN_API_KEY")
base_url_embed = "https://model-lqzx40k3.api.baseten.co/environments/production/sync"
base_url_rerank = "https://model-e3mx5vzq.api.baseten.co/environments/production/sync"
base_url_fake = "fake_url"

IS_NUMPY_AVAILABLE = False
try:
    import numpy as np

    IS_NUMPY_AVAILABLE = True
except ImportError:
    pass


def is_deployment_reachable(base_url, route="/v1/embeddings", timeout=5):
    try:
        response = requests.post(
            f"{base_url}{route}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "my_model",
                "input": ["Hello world", "Hello world 2"],
                "query": "Hello world",
                "texts": ["Hello world", "Hello world 2"],
                "inputs": [["prediction"]],
            },
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


is_deployment_reachable(base_url_embed, "/v1/embeddings", 0.1)
is_deployment_reachable(base_url_rerank, "/rerank", 0.1)
EMBEDDINGS_REACHABLE = is_deployment_reachable(base_url_embed, "/v1/embeddings")
RERANK_REACHABLE = is_deployment_reachable(base_url_rerank, "/rerank")
CLASSIFY_REACHABLE = RERANK_REACHABLE


@pytest.mark.parametrize(
    "batch_size,max_concurrent_requests", [(1, 1000), (1000, 1), (1000, 1000), (0, 0)]
)
def test_invalid_concurrency_settings_test(batch_size, max_concurrent_requests):
    client = PerformanceClient(base_url=base_url_fake, api_key=api_key)
    assert client.api_key == api_key
    with pytest.raises(ValueError) as excinfo:
        client.embed(
            ["Hello world", "Hello world 2"],
            model="my_model",
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent_requests,
        )
    assert "must be greater" in str(excinfo.value)


def test_not_nice_concurrency_settings():
    client = PerformanceClient(base_url=base_url_fake, api_key=api_key)
    assert client.api_key == api_key
    with pytest.raises(ValueError) as excinfo:
        client.embed(
            ["Hello world", "Hello world 2"],
            model="my_model",
            batch_size=1,
            max_concurrent_requests=384,
        )
    assert "be nice" in str(excinfo.value)


@pytest.mark.parametrize("method", ["embed", "rerank", "classify"])
def test_wrong_api_key(method):
    client = PerformanceClient(base_url=base_url_embed, api_key="wrong_api_key")
    assert client.api_key == "wrong_api_key"
    with pytest.raises(HTTPError) as excinfo:
        if method == "embed":
            client.embed(
                ["Hello world", "Hello world 2"] * 32,
                model="my_model",
                batch_size=1,
                max_concurrent_requests=32,
            )
        elif method == "rerank":
            client.rerank(
                query="Who let the dogs out?",
                texts=["who, who?", "Paris france"] * 32,
                batch_size=1,
                max_concurrent_requests=32,
            )
        elif method == "classify":
            client.classify(
                inputs=["who, who?", "Paris france"] * 32,
                batch_size=1,
                max_concurrent_requests=32,
            )
    assert "403 Forbidden" in str(excinfo.value)
    assert excinfo.value.args[0] == 403


@pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
@pytest.mark.parametrize("try_numpy", [True, False])
def test_baseten_performance_client_embeddings_test(try_numpy):
    client = PerformanceClient(base_url=base_url_embed, api_key=api_key)

    assert client.api_key == api_key
    response = client.embed(
        ["Hello world", "Hello world 2"],
        model="my_model",
        batch_size=1,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, OpenAIEmbeddingsResponse)
    data = response.data
    assert len(data) == 2
    assert len(data[0].embedding) > 10
    assert isinstance(data[0].embedding[0], float)
    if try_numpy:
        if IS_NUMPY_AVAILABLE:
            pytest.mark.skip("Numpy is not available")
        array = response.numpy()
        assert isinstance(array, np.ndarray)
        assert array.shape == (2, len(data[0].embedding))


@pytest.mark.skipif(
    not RERANK_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
def test_baseten_performance_client_rerank():
    client = PerformanceClient(base_url=base_url_rerank, api_key=api_key)

    assert client.api_key == api_key
    response = client.rerank(
        query="Who let the dogs out?",
        texts=["who, who?", "Paris france"],
        batch_size=2,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert response.total_time >= 0
    assert isinstance(response, RerankResponse)
    assert len(response.data) == 2


@pytest.mark.skipif(
    not CLASSIFY_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
def test_baseten_performance_client_predict():
    client = PerformanceClient(base_url=base_url_rerank, api_key=api_key)

    assert client.api_key == api_key
    response = client.classify(
        inputs=["who, who?", "Paris france", "hi", "who"],
        batch_size=2,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, ClassificationResponse)


@pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
def test_embedding_high_volume():
    client = PerformanceClient(base_url=base_url_embed, api_key=api_key)

    assert client.api_key == api_key
    n_requests = 253
    response = client.embed(
        ["Hello world"] * n_requests,
        model="my_model",
        batch_size=3,
        max_concurrent_requests=37,
    )
    assert response is not None
    assert isinstance(response, OpenAIEmbeddingsResponse)
    data = response.data
    assert response.total_time >= 0
    assert len(response.individual_request_times) >= n_requests / 3
    assert len(data) == n_requests
    assert len(data[0].embedding) > 10
    assert isinstance(data[0].embedding[0], float)


def test_embedding_high_volume_return_instant():
    api_key = "wrong"
    base_url_wrong = "https://bla.notexist"
    client = PerformanceClient(base_url=base_url_wrong, api_key=api_key)

    assert client.api_key == api_key
    t_0 = time.time()
    with pytest.raises(Exception) as excinfo:
        client.embed(
            ["Hello world"] * 1000,
            model="my_model",
            batch_size=1,
            max_concurrent_requests=1,
        )
    assert "failed" in str(excinfo.value)
    assert time.time() - t_0 < 5, (
        "Request took too long to fail seems like you didn't implement drop on first error"
    )


@pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
def test_batch_post():
    client = PerformanceClient(base_url=base_url_embed, api_key=api_key)

    assert client.api_key == api_key

    openai_request_embed = {"model": "my_model", "input": ["Hello world"]}
    length = 4
    response = client.batch_post(
        url_path="/v1/embeddings",
        payloads=[openai_request_embed] * length,
        max_concurrent_requests=1,
    )
    data = response.data
    assert data is not None
    assert len(data) == length
    assert response.total_time >= 0
    assert len(response.response_headers) == length
    assert isinstance(response.response_headers[0].get("x-baseten-request-id"), str)
    assert len(response.individual_request_times) == length
    assert sum(response.individual_request_times) < response.total_time
    assert data[0]


@pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
def test_embed_gil_release():
    client_embed = PerformanceClient(base_url=base_url_embed, api_key=api_key)

    def embed_job(start_time):
        time.sleep(0.01)
        client_embed.embed(
            ["Hello world"] * 16,
            model="my_model",
            batch_size=1,
            max_concurrent_requests=1,
        )
        return time.time() - start_time - 0.01

    embed_job(0)  # warmup
    # Measure sequential execution times (average over a few runs)
    start_t = time.time()
    seq_times = [embed_job(start_t) for _ in range(16)]

    # Run 64 embed jobs concurrently
    with ThreadPoolExecutor(max_workers=16) as executor:
        start_t = time.time()
        futures = [executor.submit(embed_job, start_t) for _ in range(16)]
        parallel_times = [f.result() for f in futures]

    # the sequential times should sum to a much greater > 4x than the parallel times sum
    # as we implment server side batching.
    # unless the gil is not released and no-thread-concurrency == sequential
    assert seq_times[-1] > 4 * max(parallel_times)


@pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
@pytest.mark.anyio
async def test_embed_async():
    client = PerformanceClient(base_url=base_url_embed, api_key=api_key)

    response = await client.async_embed(
        ["Hello world", "Hello world 2"],
        model="my_model",
        batch_size=1,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, OpenAIEmbeddingsResponse)
    data = response.data
    assert len(data) == 2
    assert len(data[0].embedding) > 10
    assert isinstance(data[0].embedding[0], float)
    print("async test passed", data[0].embedding[0])


@pytest.mark.skipif(
    not CLASSIFY_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
@pytest.mark.anyio
async def test_classify_async():
    client = PerformanceClient(base_url=base_url_rerank, api_key=api_key)

    response = await client.async_classify(
        inputs=["who, who?", "Paris france"], batch_size=2, max_concurrent_requests=2
    )
    assert response is not None
    assert isinstance(response, ClassificationResponse)
    data = response.data
    assert len(data) == 2
    assert response.total_time >= 0
    print("async test passed", data[0])


@pytest.mark.skipif(
    not RERANK_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
@pytest.mark.anyio
async def test_rerank_async():
    client = PerformanceClient(base_url=base_url_rerank, api_key=api_key)

    response = await client.async_rerank(
        query="Who let the dogs out?",
        texts=["who, who?", "Paris france"],
        batch_size=2,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, RerankResponse)
    data = response.data
    assert len(data) == 2
    print("async test passed", data[0])


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_embed_async())
    asyncio.run(test_classify_async())
    asyncio.run(test_rerank_async())
