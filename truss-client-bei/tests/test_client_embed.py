import os

import pytest
import requests
from truss_client_bei import OpenAIEmbeddingsResponse, RerankResult, SyncClient

api_key = os.environ.get("BASETEN_API_KEY")
api_base_embed = "https://model-yqv0rjjw.api.baseten.co/environments/production"
api_base_rerank = "https://model-4q9d4yx3.api.baseten.co/environments/production"


def is_deployment_reachable(api_base, route="/sync/v1/embeddings"):
    try:
        response = requests.post(
            f"{api_base}{route}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "my_model",
                "input": ["Hello world", "Hello world 2"],
                "query": "Hello world",
                "texts": ["Hello world", "Hello world 2"],
            },
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.parametrize(
    "batch_size,max_concurrent_requests", [(1, 300), (300, 1), (300, 300)]
)
def test_invalid_concurrency_settings(batch_size, max_concurrent_requests):
    client = SyncClient(api_base="https://bla.bla", api_key=api_key)
    assert client.api_key == api_key
    with pytest.raises(ValueError):
        client.embed(
            ["Hello world", "Hello world 2"],
            model="my_model",
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent_requests,
        )


@pytest.mark.skipif(
    not is_deployment_reachable(api_base_embed, "/sync/v1/embeddings"),
    reason="Deployment is not reachable. Skipping test.",
)
def test_truss_client_bei_embeddings():
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
    # assert len(response.data.embeddings) == 2
    # assert len(response.embeddings[0]) > 10
    # assert len(response.embeddings[1]) > 10
    # assert response.embeddings[0] != response.embeddings[1]
    assert len(data) == 2
    assert len(data[0].embedding) > 10


@pytest.mark.skipif(
    not is_deployment_reachable(api_base_rerank, "/sync/rerank"),
    reason="Deployment is not reachable. Skipping test.",
)
def test_truss_client_bei_rerank():
    client = SyncClient(api_base=api_base_rerank, api_key=api_key)

    assert client.api_key == api_key
    response = client.rerank(
        query="Who let the dogs out?",
        texts=["who, who?", "Paris france"],
        batch_size=1,
        max_concurrent_requests=2,
    )
    assert response is not None
    assert isinstance(response, RerankResult)
