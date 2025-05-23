# High performance client for Baseten-Embeddings

Usage with Baseten Embeddings
```
pip install truss_client_bei
```

```python
from truss_client_bei import OpenAIEmbeddingsResponse, RerankResponse, SyncClient

api_key = os.environ.get("BASETEN_API_KEY")
api_base_embed = "https://model-yqv0rjjw.api.baseten.co/environments/production"

client = SyncClient(api_base=api_base_embed, api_key=api_key)

assert client.api_key == api_key
response = client.embed(
    ["Hello world", "Hello world 2", ".."],
    model="my_model",
    # mini batch size
    batch_size=4,
    # send up to 32 mini-batch-sizes at once.
    max_concurrent_requests=32,
    # set a timeout
    timeout_s=360,
)
```

### Develop

```
apt-get install patchelf
pip install maturin[patchelf]
maturin develop
pytest tests
```
