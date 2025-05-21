import os


def test_truss_client_bei_bindings():
    from truss_client_bei import SyncClient

    SyncClient.embed


def test_truss_client_bei_embeddings():
    from truss_client_bei import SyncClient

    client = SyncClient(
        api_base="https://model-yqv0rjjw.api.baseten.co/environments/production"
    )
    assert client.api_key == os.environ["BASETEN_API_KEY"]
    response = client.embed(["Hello world"], model="my_model")
    assert response is not None
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], list)
    assert len(response[0]) > 10
    assert isinstance(response[0][0], float)
