# High performance client for Baseten.co

This library provides a high-performance client for Baseten.co endpoints including embeddings, reranking, and classification, available for both **Python/pip** and **Node.js/npm**. It was built for massive concurrent post requests to any URL, also outside of baseten.co. PerformanceClient releases the GIL while performing requests in the Rust, and supports simulaneous sync and async usage. It was benchmarked with >1200 rps per client in [our blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/). PerformanceClient is built on top of pyo3, reqwest and tokio and is MIT licensed.

![benchmarks](https://www.baseten.co/_next/image/?url=https%3A%2F%2Fwww.datocms-assets.com%2F104802%2F1749832130-diagram-9.png%3Fauto%3Dformat%26fit%3Dmax%26w%3D1200&w=3840&q=75)

## Installation

### Python
```bash
pip install baseten_performance_client
```

### Node.js
```bash
npm install baseten-performance-client
```

### Rust
```bash
cargo add baseten_performance_client_core
# Or add to your Cargo.toml:
# [dependencies]
# baseten_performance_client_core = "0.0.14"
# tokio = { version = "1.0", features = ["full"] }
```


## Usage

### Python

```python
import os
import asyncio
from baseten_performance_client import PerformanceClient, RequestProcessingPreference, OpenAIEmbeddingsResponse, RerankResponse, ClassificationResponse

api_key = os.environ.get("BASETEN_API_KEY")
base_url_embed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync"
# Also works with OpenAI or Mixedbread.
# base_url_embed = "https://api.openai.com" or "https://api.mixedbread.com"
client = PerformanceClient(base_url=base_url_embed, api_key=api_key)
```

### Node.js

```javascript
const { PerformanceClient, RequestProcessingPreference } = require('baseten-performance-client');

const apiKey = process.env.BASETEN_API_KEY;
const baseUrlEmbed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
// Also works with OpenAI or Mixedbread.
// const baseUrlEmbed = "https://api.openai.com" or "https://api.mixedbread.com"
const client = new PerformanceClient(baseUrlEmbed, apiKey);
```

## RequestProcessingPreference

All API methods accept an optional `RequestProcessingPreference` object that configures request processing behavior. This provides a clean, reusable way to configure concurrency, timeouts, and retry/hedging budgets.

### Python
```python
from baseten_performance_client import RequestProcessingPreference

# Create with defaults
pref = RequestProcessingPreference()

# Or customize all parameters
pref = RequestProcessingPreference(
    max_concurrent_requests=64,   # Max parallel requests (default: 128)
    batch_size=32,                # Items per batch (default: 128)
    timeout_s=30.0,               # Per-request timeout in seconds (default: 3600)
    max_chars_per_request=10000,  # Character-based batching limit (optional)
    hedge_delay=0.5,              # Request hedging delay in seconds (optional, min 0.2s)
    total_timeout_s=600.0,        # Total timeout for all requests (optional)
    hedge_budget_pct=0.10,        # Percentage of requests allowed for hedging (default: 10%)
    retry_budget_pct=0.05         # Percentage of requests allowed for retries (default: 5%)
)
```

### Node.js
```javascript
const { RequestProcessingPreference } = require('baseten-performance-client');

// Create with defaults
const pref = new RequestProcessingPreference();

// Or customize all parameters
const pref = new RequestProcessingPreference(
    64,     // maxConcurrentRequests
    32,     // batchSize
    30.0,   // timeoutS
    10000,  // maxCharsPerRequest (optional)
    0.5,    // hedgeDelay (optional)
    600.0,  // totalTimeoutS (optional)
    0.10,   // hedgeBudgetPct
    0.05    // retryBudgetPct
);
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent_requests` | 128 | Maximum number of concurrent HTTP requests |
| `batch_size` | 128 | Number of items per batch/request |
| `timeout_s` | 3600.0 | Timeout for individual requests in seconds |
| `max_chars_per_request` | None | Character-based batching limit (50-256,000) |
| `hedge_delay` | None | Request hedging delay in seconds (min 0.2s) |
| `total_timeout_s` | None | Total timeout for all batched requests |
| `hedge_budget_pct` | 0.10 | Percentage of total requests that can be hedged |
| `retry_budget_pct` | 0.05 | Percentage of total requests that can be retried |

### Embeddings
#### Python Embedding

```python
texts = ["Hello world", "Example text", "Another sample"]

# Simple usage with defaults
response = client.embed(input=texts, model="my_model")

# With custom preferences
pref = RequestProcessingPreference(
    batch_size=4,
    max_concurrent_requests=32,
    timeout_s=360,
    max_chars_per_request=10000,
    hedge_delay=0.5,
    total_timeout_s=600
)
response = client.embed(input=texts, model="my_model", preference=pref)

# Accessing embedding data
print(f"Model used: {response.model}")
print(f"Total tokens used: {response.usage.total_tokens}")
print(f"Total time: {response.total_time:.4f}s")
if response.individual_request_times:
    for i, batch_time in enumerate(response.individual_request_times):
        print(f"  Time for batch {i}: {batch_time:.4f}s")

for i, embedding_data in enumerate(response.data):
    print(f"Embedding for text {i} (original input index {embedding_data.index}):")
    if isinstance(embedding_data.embedding, list):
        print(f"  First 3 dimensions: {embedding_data.embedding[:3]}")
        print(f"  Length: {len(embedding_data.embedding)}")

# Using the numpy() method (requires numpy to be installed)
import numpy as np
numpy_array = response.numpy()
print(f"Embeddings as NumPy array: {numpy_array.shape}")
```

Note: The embed method is versatile and can be used with any embeddings service, e.g. OpenAI API embeddings, not just for Baseten deployments.

#### Asynchronous Embedding

```python
async def async_embed():
    texts = ["Async hello", "Async example"]
    pref = RequestProcessingPreference(
        batch_size=2,
        max_concurrent_requests=16,
        timeout_s=360,
        hedge_delay=1.5
    )
    response = await client.async_embed(input=texts, model="my_model", preference=pref)
    print("Async embedding response:", response.data)

# To run:
# asyncio.run(async_embed())
```

#### Node.js Embedding

```javascript
const texts = ["Hello world", "Example text", "Another sample"];

// Simple usage with defaults
const response = await client.embed(texts, "my_model");

// With custom preferences
const pref = new RequestProcessingPreference(32, 4, 360.0, 10000, 0.5);
const response = await client.embed(
    texts,                      // input
    "my_model",                 // model
    null,                       // encodingFormat
    null,                       // dimensions
    null,                       // user
    pref                        // preference
);

// Accessing embedding data
console.log(`Model used: ${response.model}`);
console.log(`Total tokens used: ${response.usage.total_tokens}`);
console.log(`Total time: ${response.total_time.toFixed(4)}s`);

response.data.forEach((embeddingData, i) => {
    console.log(`Embedding for text ${i} (original input index ${embeddingData.index}):`);
    console.log(`  First 3 dimensions: ${embeddingData.embedding.slice(0, 3)}`);
    console.log(`  Length: ${embeddingData.embedding.length}`);
});
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

pref = RequestProcessingPreference(max_concurrent_requests=96, timeout_s=360)
response_obj = client.batch_post(
    url_path="/v1/embeddings",
    payloads=[payload1, payload2],
    preference=pref
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
    pref = RequestProcessingPreference(max_concurrent_requests=4, timeout_s=360)
    response_obj = await client.async_batch_post(
        url_path="/v1/embeddings",
        payloads=[payload1, payload2],
        preference=pref
    )
    print(f"Async total time for batch POST: {response_obj.total_time:.4f}s")

# To run:
# asyncio.run(async_batch_post_example())
```

#### Node.js Batch POST

```javascript
const payload1 = { model: "my_model", input: ["Batch request sample 1"] };
const payload2 = { model: "my_model", input: ["Batch request sample 2"] };

const pref = new RequestProcessingPreference(96, null, 360.0);
const responseObj = await client.batchPost(
    "/v1/embeddings",           // urlPath
    [payload1, payload2],       // payloads
    pref                        // preference
);

console.log(`Total time for batch POST: ${responseObj.total_time.toFixed(4)}s`);
responseObj.data.forEach((respData, i) => {
    console.log(`Response ${i + 1}:`);
    console.log(`  Data:`, respData);
    console.log(`  Headers:`, responseObj.response_headers[i]);
    console.log(`  Time taken: ${responseObj.individual_request_times[i].toFixed(4)}s`);
});
```
### Reranking
Reranking compatible with BEI or text-embeddings-inference.

#### Synchronous Reranking

```python
query = "What is the best framework?"
documents = ["Doc 1 text", "Doc 2 text", "Doc 3 text"]

pref = RequestProcessingPreference(
    batch_size=2,
    max_concurrent_requests=16,
    timeout_s=360,
    max_chars_per_request=5000,
    hedge_delay=1.5
)
rerank_response = client.rerank(
    query=query,
    texts=documents,
    return_text=True,
    preference=pref
)
for res in rerank_response.data:
    print(f"Index: {res.index} Score: {res.score}")
```

#### Asynchronous Reranking

```python
async def async_rerank():
    query = "Async query sample"
    docs = ["Async doc1", "Async doc2"]
    pref = RequestProcessingPreference(batch_size=1, max_concurrent_requests=8, timeout_s=360)
    response = await client.async_rerank(
        query=query,
        texts=docs,
        return_text=True,
        preference=pref
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

pref = RequestProcessingPreference(
    batch_size=2,
    max_concurrent_requests=16,
    timeout_s=360.0,
    max_chars_per_request=6000,
    hedge_delay=10.0
)
classify_response = client.classify(inputs=texts_to_classify, preference=pref)
for group in classify_response.data:
    for result in group:
        print(f"Label: {result.label}, Score: {result.score}")
```

#### Asynchronous Classification
```python
async def async_classify():
    texts = ["Async positive", "Async negative"]
    pref = RequestProcessingPreference(batch_size=1, max_concurrent_requests=8, timeout_s=360)
    response = await client.async_classify(inputs=texts, preference=pref)
    for group in response.data:
        for res in group:
            print(f"Async Label: {res.label}, Score: {res.score}")

# To run:
# asyncio.run(async_classify())
```


### Error Handling

The client can raise several types of errors. Here's how to handle common ones:

- **`requests.exceptions.HTTPError`**: This error is raised for HTTP issues, such as authentication failures (e.g., 403 Forbidden if the API key is wrong), server errors (e.g., 5xx), or if the endpoint is not found (404). You can inspect `e.response.status_code` and `e.response.text` (or `e.response.json()` if the body is JSON) for more details.
- **`requests.exceptions.Timeout`**: This error is raised when a request or the total operation times out (based on `timeout_s` or `total_timeout_s`).
- **`ValueError`**: This error can occur due to invalid input parameters (e.g., an empty `input` list for `embed`, invalid `batch_size` or `max_concurrent_requests` values). It can also be raised by `response.numpy()` if embeddings are not float vectors or have inconsistent dimensions.

Here's an example demonstrating how to catch these errors for the `embed` method:

```python
import requests

# client = PerformanceClient(base_url="your_b10_url", api_key="your_b10_api_key")

texts_to_embed = ["Hello world", "Another text example"]
try:
    pref = RequestProcessingPreference(batch_size=2, max_concurrent_requests=4, timeout_s=60)
    response = client.embed(
        input=texts_to_embed,
        model="your_embedding_model",
        preference=pref
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


## Rust

```rust
use baseten_performance_client_core::{PerformanceClientCore, RequestProcessingPreference, ClientError};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("BASETEN_API_KEY").expect("BASETEN_API_KEY not set");
    let base_url = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";

    let client = PerformanceClientCore::new(base_url.to_string(), Some(api_key), 1, None)?;

    // Create preference with builder pattern
    let pref = RequestProcessingPreference::new()
        .with_batch_size(4)
        .with_max_concurrent_requests(32)
        .with_timeout_s(360.0)
        .with_max_chars_per_request(Some(10000))
        .with_hedge_delay(Some(0.5));

    // Embedding example
    let texts = vec!["Hello world".to_string(), "Example text".to_string()];
    let (embedding_response, durations, headers, total_time) = client
        .process_embeddings_requests(
            texts,
            "my_model".to_string(),
            None,  // encoding_format
            None,  // dimensions
            None,  // user
            &pref,
        )
        .await?;

    println!("Model: {}", embedding_response.model);
    println!("Total tokens: {}", embedding_response.usage.total_tokens);
    println!("Total time: {:.4}s", total_time.as_secs_f64());

    // Batch POST example
    let payloads = vec![
        serde_json::json!({"model": "my_model", "input": ["Rust sample 1"]}),
        serde_json::json!({"model": "my_model", "input": ["Rust sample 2"]}),
    ];

    let batch_pref = RequestProcessingPreference::new()
        .with_max_concurrent_requests(32)
        .with_timeout_s(360.0);

    let (results, batch_total_time) = client
        .process_batch_post_requests(
            "/v1/embeddings".to_string(),
            payloads,
            &batch_pref,
            None,  // custom_headers
        )
        .await?;

    println!("Batch POST total time: {:.4}s", batch_total_time.as_secs_f64());

    Ok(())
}
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
