import os
import time

import pytest
import requests
from bei_client import (
    ClassificationResponse,
    OpenAIEmbeddingsResponse,
    RerankResponse,
    SyncClient,
)

api_key = os.environ.get("BASETEN_API_KEY")
api_base_embed = "https://model-yqv0rjjw.api.baseten.co/environments/production/sync"
api_base_rerank = "https://model-4q9d4yx3.api.baseten.co/environments/production/sync"
api_base_fake = "fake_url"

IS_NUMPY_AVAILABLE = False
try:
    import numpy as np

    IS_NUMPY_AVAILABLE = True
except ImportError:
    pass


def is_deployment_reachable(api_base, route="/v1/embeddings", timeout=5):
    try:
        response = requests.post(
            f"{api_base}{route}",
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


is_deployment_reachable(api_base_embed, "/v1/embeddings", 0.1)
is_deployment_reachable(api_base_rerank, "/rerank", 0.1)
EMBEDDINGS_REACHABLE = is_deployment_reachable(api_base_embed, "/v1/embeddings")
RERANK_REACHABLE = is_deployment_reachable(api_base_rerank, "/rerank")
CLASSIFY_REACHABLE = RERANK_REACHABLE


@pytest.mark.parametrize(
    "batch_size,max_concurrent_requests", [(1, 1000), (1000, 1), (1000, 1000), (0, 0)]
)
def test_invalid_concurrency_settings_test(batch_size, max_concurrent_requests):
    client = SyncClient(api_base=api_base_fake, api_key=api_key)
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
    client = SyncClient(api_base=api_base_fake, api_key=api_key)
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
    client = SyncClient(api_base=api_base_embed, api_key="wrong_api_key")
    assert client.api_key == "wrong_api_key"
    with pytest.raises(Exception) as excinfo:
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


@pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
@pytest.mark.parametrize("try_numpy", [True, False])
def test_bei_client_embeddings_test(try_numpy):
    client = SyncClient(api_base=api_base_embed, api_key=api_key)

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
def test_bei_client_rerank():
    client = SyncClient(api_base=api_base_rerank, api_key=api_key)

    assert client.api_key == api_key
    response = client.rerank(
        query="Who let the dogs out?",
        texts=["who, who?", "Paris france"],
        batch_size=2,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, RerankResponse)
    assert len(response.data) == 2


@pytest.mark.skipif(
    not CLASSIFY_REACHABLE, reason="Deployment is not reachable. Skipping test."
)
def test_bei_client_predict():
    client = SyncClient(api_base=api_base_rerank, api_key=api_key)

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
    client = SyncClient(api_base=api_base_embed, api_key=api_key)

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
    assert len(data) == n_requests
    assert len(data[0].embedding) > 10
    assert isinstance(data[0].embedding[0], float)


def test_embedding_high_volume_return_instant():
    api_key = "wrong"
    api_base_wrong = "https://bla.notexist"
    client = SyncClient(api_base=api_base_wrong, api_key=api_key)

    assert client.api_key == api_key
    t_0 = time.time()
    with pytest.raises(Exception) as excinfo:
        client.embed(
            ["Hello world"] * 10,
            model="my_model",
            batch_size=1,
            max_concurrent_requests=1,
        )
    assert "failed" in str(excinfo.value)
    assert time.time() - t_0 < 5, (
        "Request took too long to fail seems like you didn't implement drop on first error"
    )
