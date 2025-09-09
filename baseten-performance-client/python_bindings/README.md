# High performance client for Baseten.co

This library provides a high-performance Python client for Baseten.co endpoints including embeddings, reranking, and classification. It was built for massive concurrent post requests to any URL, also outside of baseten.co. PerformanceClient releases the GIL while performing requests in the Rust, and supports simultaneous sync and async usage. It was benchmarked with >1200 rps per client in [our blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/). PerformanceClient is built on top of pyo3, reqwest and tokio and is MIT licensed.

![benchmarks](https://www.baseten.co/_next/image/?url=https%3A%2F%2Fwww.datocms-assets.com%2F104802%2F1749832130-diagram-9.png%3Fauto%3Dformat%26fit%3Dmax%26w%3D1200&w=3840&q=75)

## Installation

```
pip install baseten_performance_client
```

## Usage

```python
import os
import asyncio
from baseten_performance_client import PerformanceClient, OpenAIEmbeddingsResponse, RerankResponse, ClassificationResponse

api_key = os.environ.get("BASETEN_API_KEY")
base_url_embed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync"
# Also works with OpenAI or Mixedbread.
# base_url_embed = "https://api.openai.com" or "https://api.mixedbread.com"
client = PerformanceClient(base_url=base_url_embed, api_key=api_key)
```
### Embeddings
#### Synchronous Embedding

```python
texts = ["Hello world", "Example text", "Another sample"]
response = client.embed(
    input=texts,
    model="my_model",
    batch_size=4,
    max_concurrent_requests=32,
    timeout_s=360
)

# Accessing embedding data
print(f"Model used: {response.model}")
print(f"Total tokens used: {response.usage.total_tokens}")
print(f"Total time: {response.total_time:.4f}s")
if response.individual_batch_request_times:
    for i, batch_time in enumerate(response.individual_batch_request_times):
        print(f"  Time for batch {i}: {batch_time:.4f}s")

for i, embedding_data in enumerate(response.data):
    print(f"Embedding for text {i} (original input index {embedding_data.index}):")
    # embedding_data.embedding can be List[float] or str (base64)
    if isinstance(embedding_data.embedding, list):
        print(f"  First 3 dimensions: {embedding_data.embedding[:3]}")
        print(f"  Length: {len(embedding_data.embedding)}")

# Using the numpy() method (requires numpy to be installed)
import numpy as np
numpy_array = response.numpy()
print("\nEmbeddings as NumPy array:")
print(f"  Shape: {numpy_array.shape}")
print(f"  Data type: {numpy_array.dtype}")
if numpy_array.shape[0] > 0:
    print(f"  First 3 dimensions of the first embedding: {numpy_array[0][:3]}")

```

Note: The embed method is versatile and can be used with any embeddings service, e.g. OpenAI API embeddings, not just for Baseten deployments.

#### Asynchronous Embedding

```python
async def async_embed():
    texts = ["Async hello", "Async example"]
    response = await client.async_embed(
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

#### Embedding Benchmarks
Comparison against `pip install openai` for `/v1/embeddings`. Tested with the `./scripts/compare_latency_openai.py` with mini_batch_size of 128, and 4 server-side replicas. Results with OpenAI similar, OpenAI allows a max mini_batch_size of 2048.

| Number of inputs / embeddings | Number of Tasks | PerformanceClient (s) | AsyncOpenAI (s) | Speedup |
|-------------------------------:|---------------:|---------------------:|----------------:|--------:|
| 128                            |              1 |                0.12 |            0.13 |    1.08× |
| 512                            |              4 |                0.14 |            0.21 |    1.50× |
| 8 192                          |             64 |                0.83 |            1.95 |    2.35× |
| 131 072                        |           1 024 |                4.63 |           39.07 |    8.44× |
| 2 097 152                      |          16 384 |               70.92 |          903.68 |   12.74× |

### General Batch POST

The batch_post method is generic. It can be used to send POST requests to any URL, not limited to Baseten endpoints. The input and output can be any JSON item.

#### Synchronous Batch POST
```python
payload1 = {"model": "my_model", "input": ["Batch request sample 1"]}
payload2 = {"model": "my_model", "input": ["Batch request sample 2"]}
response_obj = client.batch_post(
    url_path="/v1/embeddings", # Example path, adjust to your needs
    payloads=[payload1, payload2],
    max_concurrent_requests=96,
    timeout_s=360
)
print(f"Total time for batch POST: {response_obj.total_time:.4f}s")
for i, (resp_data, headers, time_taken) in enumerate(zip(response_obj.data, response_obj.response_headers, response_obj.individual_request_times)):
    print(f"Response {i+1}:")
    print(f"  Data: {resp_data}")
    print(f"  Headers: {headers}")
    print(f"  Time taken: {time_taken:.4f}s")
```

#### Asynchronous Batch POST

```python
async def async_batch_post_example():
    payload1 = {"model": "my_model", "input": ["Async batch sample 1"]}
    payload2 = {"model": "my_model", "input": ["Async batch sample 2"]}
    response_obj = await client.async_batch_post(
        url_path="/v1/embeddings",
        payloads=[payload1, payload2],
        max_concurrent_requests=4,
        timeout_s=360
    )
    print(f"Async total time for batch POST: {response_obj.total_time:.4f}s")
    for i, (resp_data, headers, time_taken) in enumerate(zip(response_obj.data, response_obj.response_headers, response_obj.individual_request_times)):
        print(f"Async Response {i+1}:")
        print(f"  Data: {resp_data}")
        print(f"  Headers: {headers}")
        print(f"  Time taken: {time_taken:.4f}s")

# To run:
# asyncio.run(async_batch_post_example())
```
### Reranking
Reranking compatible with BEI or text-embeddings-inference.

#### Synchronous Reranking

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

#### Asynchronous Reranking

```python
async def async_rerank():
    query = "Async query sample"
    docs = ["Async doc1", "Async doc2"]
    response = await client.async_rerank(
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

### Classification
Predict (classification endpoint) compatible with BEI or text-embeddings-inference.
#### Synchronous Classification

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

#### Asynchronous Classification
```python
async def async_classify():
    texts = ["Async positive", "Async negative"]
    response = await client.async_classify(
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


### Error Handling

The client can raise several types of errors. Here's how to handle common ones:

- **`requests.exceptions.HTTPError`**: This error is raised for HTTP issues, such as authentication failures (e.g., 403 Forbidden if the API key is wrong), server errors (e.g., 5xx), or if the endpoint is not found (404). You can inspect `e.response.status_code` and `e.response.text` (or `e.response.json()` if the body is JSON) for more details.
- **`ValueError`**: This error can occur due to invalid input parameters (e.g., an empty `input` list for `embed`, invalid `batch_size` or `max_concurrent_requests` values). It can also be raised by `response.numpy()` if embeddings are not float vectors or have inconsistent dimensions.

Here's an example demonstrating how to catch these errors for the `embed` method:

```python
import requests

# client = PerformanceClient(base_url="your_baseten_url", api_key="your_baseten_api_key")

texts_to_embed = ["Hello world", "Another text example"]
try:
    response = client.embed(
        input=texts_to_embed,
        model="your_embedding_model", # Replace with your actual model name
        batch_size=2,
        max_concurrent_requests=4,
        timeout_s=60 # Timeout in seconds
    )
    # Process successful response
    print(f"Model used: {response.model}")
    print(f"Total tokens: {response.usage.total_tokens}")
    for item in response.data:
        embedding_preview = item.embedding[:3] if isinstance(item.embedding, list) else "Base64 Data"
        print(f"Index {item.index}, Embedding (first 3 dims or type): {embedding_preview}")

except requests.exceptions.HTTPError as e:
    print(f"An HTTP error occurred: {e}, code {e.args[0]}")

```

For asynchronous methods (`async_embed`, `async_rerank`, `async_classify`, `async_batch_post`), the same exceptions will be raised by the `await` call and can be caught using a `try...except` block within an `async def` function.

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
