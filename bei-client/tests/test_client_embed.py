import os

import pytest
import requests
from bei_client import (
    ClassificationResponse,
    OpenAIEmbeddingsResponse,
    RerankResponse,
    SyncClient,
)

api_key = os.environ.get("BASETEN_API_KEY")
api_base_embed = "https://model-yqv0rjjw.api.baseten.co/environments/production"
api_base_rerank = "https://model-4q9d4yx3.api.baseten.co/environments/production"


def is_deployment_reachable(api_base, route="/sync/v1/embeddings", timeout=5):
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


is_deployment_reachable(api_base_embed, "/sync/v1/embeddings", 0.1)
is_deployment_reachable(api_base_rerank, "/sync/rerank", 0.1)


@pytest.mark.parametrize(
    "batch_size,max_concurrent_requests", [(1, 300), (300, 1), (300, 300)]
)
def test_invalid_concurrency_settings_test(batch_size, max_concurrent_requests):
    client = SyncClient(api_base="https://bla.bla", api_key=api_key)
    assert client.api_key == api_key
    with pytest.raises(ValueError):
        client.embed(
            ["Hello world", "Hello world 2"],
            model="my_model",
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent_requests,
        )


@pytest.mark.parametrize("method", ["embed", "rerank", "classify"])
def test_wrong_api_key(method):
    client = SyncClient(api_base=api_base_embed, api_key="wrong_api_key")
    assert client.api_key == "wrong_api_key"
    with pytest.raises(Exception) as excinfo:
        if method == "embed":
            client.embed(
                ["Hello world", "Hello world 2"],
                model="my_model",
                batch_size=1,
                max_concurrent_requests=2,
            )
        elif method == "rerank":
            client.rerank(
                query="Who let the dogs out?",
                texts=["who, who?", "Paris france"],
                batch_size=2,
                max_concurrent_requests=2,
            )
        elif method == "classify":
            client.classify(
                inputs=["who, who?", "Paris france", "hi", "who", "who?"],
                batch_size=2,
                max_concurrent_requests=2,
            )
    assert "403 Forbidden" in str(excinfo.value)


@pytest.mark.skipif(
    not is_deployment_reachable(api_base_embed, "/sync/v1/embeddings"),
    reason="Deployment is not reachable. Skipping test.",
)
def test_bei_client_embeddings_test():
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


@pytest.mark.skipif(
    not is_deployment_reachable(api_base_rerank, "/sync/rerank"),
    reason="Deployment is not reachable. Skipping test.",
)
def bei_client_rerank_test():
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
    not is_deployment_reachable(api_base_rerank, "/sync/predict"),
    reason="Deployment is not reachable. Skipping test.",
)
def bei_client_predict_test():
    client = SyncClient(api_base=api_base_rerank, api_key=api_key)

    assert client.api_key == api_key
    response = client.classify(
        inputs=["who, who?", "Paris france", "hi", "who", "who?"],
        batch_size=2,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, ClassificationResponse)
    assert len(response.data) == 5


@pytest.mark.skipif(
    not is_deployment_reachable(api_base_embed, "/sync/v1/embeddings"),
    reason="Deployment is not reachable. Skipping test.",
)
def embedding_high_volume_test():
    client = SyncClient(api_base=api_base_embed, api_key=api_key)

    assert client.api_key == api_key
    n_requests = 200
    response = client.embed(
        ["Hello world"] * n_requests,
        model="my_model",
        batch_size=1,
        max_concurrent_requests=512,
    )
    assert response is not None
    assert isinstance(response, OpenAIEmbeddingsResponse)
    data = response.data
    assert len(data) == n_requests
    assert len(data[0].embedding) > 10
    assert isinstance(data[0].embedding[0], float)
