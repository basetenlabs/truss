import os

import pytest
import requests
from truss_client_bei import OpenAIEmbeddingsResponse, SyncClient

api_key = os.environ.get("BASETEN_API_KEY")
api_base = "https://model-yqv0rjjw.api.baseten.co/environments/production"


def is_deployment_reachable():
    try:
        response = requests.post(
            f"{api_base}/sync/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "my_model", "input": ["Hello world", "Hello world 2"]},
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(
    not is_deployment_reachable(), reason="Deployment is not reachable. Skipping test."
)
def test_truss_client_bei_embeddings():
    api_key = os.environ["BASETEN_API_KEY"]
    client = SyncClient(
        api_base="https://model-yqv0rjjw.api.baseten.co/environments/production",
        api_key=api_key,
    )

    assert client.api_key == os.environ["BASETEN_API_KEY"]
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
