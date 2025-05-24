# High performance client for Baseten.co

This library provides a high-performance Python client for Baseten.co endpoints including embeddings, reranking, and classification. It supports both synchronous and asynchronous operations. It also supports parallel post requests to any URL, also outside of baseten.co. InferenceClient releases the GIL while performing requests in the background.

## Installation

```
pip install baseten_inference_client
```

## Usage

Import the client and set your API key. Note that we now use the new endpoint URL with `/sync`.

```python
import os
import asyncio
from baseten_inference_client import InferenceClient, OpenAIEmbeddingsResponse, RerankResponse, ClassificationResponse

api_key = os.environ.get("BASETEN_API_KEY")
api_base_embed = "https://model-yqv0rjjw.api.baseten.co/environments/production/sync"

client = InferenceClient(api_base=api_base_embed, api_key=api_key)
```

### Synchronous Embedding

```python
texts = ["Hello world", "Example text", "Another sample"]
response = client.embed(
    input=texts,
    model="my_model",
    batch_size=4,
    max_concurrent_requests=32,
    timeout_s=360
)
print("Embedding response:", response.data)
```

Note: The embed method is versatile and can be used with any embeddings service, e.g. OpenAI API embeddings, not just for Baseten deployments.

### Asynchronous Embedding

```python
async def async_embed():
    texts = ["Async hello", "Async example"]
    response = await client.aembed(
        input=texts,
        model="my_model",
        batch_size=2,
        max_concurrent_requests=16,
        timeout_s=360
    )
    print("Async embedding response:", response.data)

# To run:
# asyncio.run(async_embed())
```

### Synchronous Batch POST

```python
payload = {"model": "my_model", "input": ["Batch request sample"]}
batch_response = client.batch_post(
    url_path="/v1/embeddings",
    payloads=[payload, payload],
    max_concurrent_requests=4,
    timeout_s=360
)
print("Batch POST responses:", batch_response)
```

Note: The batch_post method is generic. It can be used to send POST requests to any URL,
not limited to Baseten endpoints.

### Asynchronous Batch POST

```python
async def async_batch_post():
    payload = {"model": "my_model", "input": ["Async batch sample"]}
    responses = await client.abatch_post(
        url_path="/v1/embeddings",
        payloads=[payload, payload],
        max_concurrent_requests=4,
        timeout_s=360
    )
    print("Async batch POST responses:", responses)

# To run:
# asyncio.run(async_batch_post())
```

### Synchronous Reranking

```python
query = "What is the best framework?"
documents = ["Doc 1 text", "Doc 2 text", "Doc 3 text"]
rerank_response = client.rerank(
    query=query,
    texts=documents,
    return_text=True,
    batch_size=2,
    max_concurrent_requests=16,
    timeout_s=360
)
for res in rerank_response.data:
    print(f"Index: {res.index} Score: {res.score}")
```

### Asynchronous Reranking

```python
async def async_rerank():
    query = "Async query sample"
    docs = ["Async doc1", "Async doc2"]
    response = await client.arerank(
        query=query,
        texts=docs,
        return_text=True,
        batch_size=1,
        max_concurrent_requests=8,
        timeout_s=360
    )
    for res in response.data:
        print(f"Async Index: {res.index} Score: {res.score}")

# To run:
# asyncio.run(async_rerank())
```

### Synchronous Classification

```python
texts_to_classify = [
    "This is great!",
    "I did not like it.",
    "Neutral experience."
]
classify_response = client.classify(
    inputs=texts_to_classify,
    batch_size=2,
    max_concurrent_requests=16,
    timeout_s=360
)
for group in classify_response.data:
    for result in group:
        print(f"Label: {result.label}, Score: {result.score}")
```

### Asynchronous Classification

```python
async def async_classify():
    texts = ["Async positive", "Async negative"]
    response = await client.aclassify(
        inputs=texts,
        batch_size=1,
        max_concurrent_requests=8,
        timeout_s=360
    )
    for group in response.data:
        for res in group:
            print(f"Async Label: {res.label}, Score: {res.score}")

# To run:
# asyncio.run(async_classify())
```


## Development

```bash
# Install prerequisites
sudo apt-get install patchelf
# Install cargo if not already installed.

# Set up a Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install maturin[patchelf] pytest requests numpy

# Build and install the Rust extension in development mode
maturin develop
cargo fmt
# Run tests
pytest tests
```

## Contributions
Feel free to contribute to this repo, tag @michaelfeil for review.

## License
MIT License
