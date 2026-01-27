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
# baseten_performance_client_core = "0.0.16"
# tokio = { version = "1.0", features = ["full"] }
```


## Usage

### Python

```python
import os
import asyncio
from baseten_performance_client import PerformanceClient, OpenAIEmbeddingsResponse, RerankResponse, ClassificationResponse

api_key = os.environ.get("BASETEN_API_KEY")
base_url_embed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync"
# Also works with OpenAI or Mixedbread.
# base_url_embed = "https://api.openai.com" or "https://api.mixedbread.com"
# Basic client setup
client = PerformanceClient(base_url=base_url_embed, api_key=api_key)

# Advanced setup with HTTP version selection and connection pooling
from baseten_performance_client import HttpClientWrapper
http_wrapper = HttpClientWrapper(http_version=1)  # HTTP/1.1 (default)
advanced_client = PerformanceClient(
    base_url=base_url_embed,
    api_key=api_key,
    http_version=2,  # HTTP/2
    client_wrapper=http_wrapper  # Share connection pool
)
```

### Node.js

```javascript
const { PerformanceClient } = require('baseten-performance-client');

const apiKey = process.env.BASETEN_API_KEY;
const baseUrlEmbed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
// Also works with OpenAI or Mixedbread.
// const baseUrlEmbed = "https://api.openai.com" or "https://api.mixedbread.com"
// Basic client setup
const client = new PerformanceClient(baseUrlEmbed, apiKey);

// Advanced setup with HTTP version selection and connection pooling
const { PerformanceClient, HttpClientWrapper } = require('baseten-performance-client');
const httpWrapper = new HttpClientWrapper(2); // HTTP/2
const advancedClient = new PerformanceClient(baseUrlEmbed, apiKey, 2, httpWrapper);
```
### Embeddings
#### Python Embedding

```python
from baseten_performance_client import RequestProcessingPreference

texts = ["Hello world", "Example text", "Another sample"]
preference = RequestProcessingPreference(
    batch_size=4,
    max_concurrent_requests=32,
    timeout_s=360,
    max_chars_per_request=10000,  # Character-based batching (50-256,000)
    hedge_delay=0.5,  # Request hedging delay in seconds (min 0.2s)
    total_timeout_s=600  # Total timeout for all batched requests
)
response = client.embed(
    input=texts,
    model="my_model",
    preference=preference
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

#### Advanced Parameters

- **`max_chars_per_request`**: Character-based batching limit (50-256,000 characters). When set, requests are batched by character count rather than just input count, helping optimize for services with character-based pricing or processing limits.
- **`hedge_delay`**: Request hedging delay in seconds (minimum 0.2s). Enables sending duplicate requests after a delay to improve latency if the original request is slow. Limited by a 5% budget to prevent excessive resource usage.
- **`total_timeout_s`**: Total timeout for the entire operation in seconds. Unlike `timeout_s` (which is per-request), this sets an upper bound on the total time for all batched requests combined. Must be >= `timeout_s` if both are set. If not set, there is no upper bound on the total time for all batched requests.

#### Asynchronous Embedding

```python
async def async_embed():
    from baseten_performance_client import RequestProcessingPreference

    texts = ["Async hello", "Async example"]
    preference = RequestProcessingPreference(
        batch_size=2,
        max_concurrent_requests=16,
        timeout_s=360,
        max_chars_per_request=8000,
        hedge_delay=1.5
    )
    response = await client.async_embed(
        input=texts,
        model="my_model",
        preference=preference
    )
    print("Async embedding response:", response.data)

# To run:
# asyncio.run(async_embed())
```

#### Node.js Embedding

```javascript
const { RequestProcessingPreference } = require('baseten-performance-client');

// All methods in Node.js are async and return Promises
const texts = ["Hello world", "Example text", "Another sample"];
const preference = new RequestProcessingPreference(
    32,        // maxConcurrentRequests
    4,         // batchSize
    10000,     // maxCharsPerRequest
    360.0,     // timeoutS
    0.5        // hedgeDelay
);
const response = await client.embed(
    texts,                      // input
    "my_model",                 // model
    null,                       // encodingFormat
    null,                       // dimensions
    null,                       // user
    preference                  // preference parameter
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

The batch_post method is generic. It can be used to send HTTP requests to any URL, not limited to Baseten endpoints. The input and output can be any JSON item. Supports multiple HTTP methods (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS) with the `method` parameter.

#### Synchronous Batch POST
```python
from baseten_performance_client import RequestProcessingPreference

payload1 = {"model": "my_model", "input": ["Batch request sample 1"]}
payload2 = {"model": "my_model", "input": ["Batch request sample 2"]}
preference = RequestProcessingPreference(
    max_concurrent_requests=32,
    timeout_s=360,
    hedge_delay=0.5,  # Enable hedging with 0.5s delay
    total_timeout_s=360  # Total operation timeout
)
response_obj = client.batch_post(
    url_path="/v1/embeddings", # Example path, adjust to your needs
    payloads=[payload1, payload2],
    custom_headers={"x-custom-header": "value"},  # Custom headers
    preference=preference,
    method="POST"  # HTTP method: GET, POST, PUT, PATCH, DELETE (default: POST)
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
    from baseten_performance_client import RequestProcessingPreference

    payload1 = {"model": "my_model", "input": ["Async batch sample 1"]}
    payload2 = {"model": "my_model", "input": ["Async batch sample 2"]}
    preference = RequestProcessingPreference(
        max_concurrent_requests=32,
        timeout_s=360,
        hedge_delay=0.5,  # Enable hedging with 0.5s delay
        total_timeout_s=360  # Total operation timeout
    )
response_obj = await client.async_batch_post(
    url_path="/v1/embeddings",
    payloads=[payload1, payload2],
    custom_headers={"x-custom-header": "value"},  # Custom headers
    preference=preference,
    method="POST"  # HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS (default: POST)
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

#### Node.js Batch POST

```javascript
const { RequestProcessingPreference } = require('baseten-performance-client');

const payload1 = { model: "my_model", input: ["Batch request sample 1"] };
const payload2 = { model: "my_model", input: ["Batch request sample 2"] };
const preference = new RequestProcessingPreference(
    32,        // maxConcurrentRequests
    undefined, // batchSize
    undefined, // maxCharsPerRequest
    360.0,     // timeoutS
    0.5,       // hedgeDelay
    360.0      // totalTimeoutS
);
const responseObj = await client.batchPost(
    "/v1/embeddings",           // urlPath
    [payload1, payload2],       // payloads
    preference,                 // preference parameter
    {"x-custom-header": "value"}, // custom headers
    "POST"                      // HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS (default: POST)
);
const responseObj = await client.batchPost(
    "/v1/embeddings",           // urlPath
    [payload1, payload2],       // payloads
    undefined, undefined, undefined, undefined, undefined, // individual parameters
    {"x-custom-header": "value"}, // customHeaders
    preference                  // preference parameter
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
from baseten_performance_client import RequestProcessingPreference

query = "What is the best framework?"
documents = ["Doc 1 text", "Doc 2 text", "Doc 3 text"]
preference = RequestProcessingPreference(
    batch_size=16,
    max_concurrent_requests=32,
    timeout_s=360,
    max_chars_per_request=256000,
    hedge_delay=0.5,
    total_timeout_s=360
)
rerank_response = client.rerank(
    query=query,
    texts=documents,
    model="rerank-model",  # Optional model specification
    return_text=True,
    preference=preference
)
for res in rerank_response.data:
    print(f"Index: {res.index} Score: {res.score}")
```

#### Asynchronous Reranking

```python
async def async_rerank():
    from baseten_performance_client import RequestProcessingPreference

    query = "Async query sample"
    docs = ["Async doc1", "Async doc2"]
    preference = RequestProcessingPreference(
        batch_size=16,
        max_concurrent_requests=32,
        timeout_s=360,
        max_chars_per_request=256000,
        hedge_delay=0.5,
        total_timeout_s=360
    )
    response = await client.async_rerank(
        query=query,
        texts=docs,
        model="rerank-model",  # Optional model specification
        return_text=True,
        preference=preference
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
from baseten_performance_client import RequestProcessingPreference

texts_to_classify = [
    "This is great!",
    "I did not like it.",
    "Neutral experience."
]
preference = RequestProcessingPreference(
    batch_size=16,
    max_concurrent_requests=32,
    timeout_s=360.0,
    max_chars_per_request=256000,
    hedge_delay=0.5,
    total_timeout_s=360
)
classify_response = client.classify(
    inputs=texts_to_classify,
    model="classification-model",  # Optional model specification
    preference=preference
)
for group in classify_response.data:
    for result in group:
        print(f"Label: {result.label}, Score: {result.score}")
```

#### Asynchronous Classification
```python
async def async_classify():
    from baseten_performance_client import RequestProcessingPreference

    texts = ["Async positive", "Async negative"]
    preference = RequestProcessingPreference(
        batch_size=16,
        max_concurrent_requests=32,
        timeout_s=360,
        max_chars_per_request=256000,
        hedge_delay=0.5,
        total_timeout_s=360
    )
    response = await client.async_classify(
        inputs=texts,
        model="classification-model",  # Optional model specification
        preference=preference
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
- **`requests.exceptions.Timeout`**: This error is raised when a request or the total operation times out (based on `timeout_s` or `total_timeout_s`).
- **`ValueError`**: This error can occur due to invalid input parameters (e.g., an empty `input` list for `embed`, invalid `batch_size` or `max_concurrent_requests` values). It can also be raised by `response.numpy()` if embeddings are not float vectors or have inconsistent dimensions.

Here's an example demonstrating how to catch these errors for the `embed` method:

```python
import requests

# client = PerformanceClient(base_url="your_b10_url", api_key="your_b10_api_key")

texts_to_embed = ["Hello world", "Another text example"]
from baseten_performance_client import RequestProcessingPreference

try:
    preference = RequestProcessingPreference(
        batch_size=2,
        max_concurrent_requests=4,
        timeout_s=60 # Timeout in seconds
    )
    response = client.embed(
        input=texts_to_embed,
        model="your_embedding_model", # Replace with your actual model name
        preference=preference
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

### Advanced Features

#### RequestProcessingPreference

The `RequestProcessingPreference` class provides a unified way to configure all request processing parameters. This is the recommended approach for advanced configuration as it provides better type safety and clearer intent.

```python
from baseten_performance_client import RequestProcessingPreference

# Create a preference with custom settings
preference = RequestProcessingPreference(
    max_concurrent_requests=64,        # Parallel requests (default: 128)
    batch_size=32,                     # Items per batch (default: 128)
    timeout_s=30.0,                   # Per-request timeout (default: 3600.0)
    hedge_delay=0.5,                  # Hedging delay (default: None)
    hedge_budget_pct=0.15,            # Hedge budget percentage (default: 0.10)
    retry_budget_pct=0.08,            # Retry budget percentage (default: 0.05)
    total_timeout_s=300.0              # Total operation timeout (default: None)
)

# Use with any method
response = client.embed(
    input=["text1", "text2"],
    model="my_model",
    preference=preference
)
```

```javascript
const { RequestProcessingPreference } = require('baseten-performance-client');

// Create a preference with custom settings
const preference = new RequestProcessingPreference(
    64,        // maxConcurrentRequests (default: 128)
    32,        // batchSize (default: 128)
    undefined, // maxCharsPerRequest
    30.0,      // timeoutS (default: 3600.0)
    0.5,       // hedgeDelay
    undefined, // totalTimeoutS
    0.15,      // hedgeBudgetPct (default: 0.10)
    0.08       // retryBudgetPct (default: 0.05)
);

// Use with any method
const response = await client.embed(
    ["text1", "text2"],
    "my_model",
    undefined, undefined, undefined, // encodingFormat, dimensions, user
    preference // preference parameter
);
```

**Budget Percentages:**
- `hedge_budget_pct`: Percentage of total requests allocated for hedging (default: 10%)
- `retry_budget_pct`: Percentage of total requests allocated for retries (default: 5%)
- Maximum allowed: 300% for both budgets

#### HTTP Version Selection
Choose between HTTP/1.1 and HTTP/2 for optimal performance:

```python
# HTTP/1.1 (default, better for high concurrency)
client_http1 = PerformanceClient(base_url, api_key, http_version=1)

# HTTP/2 (better for single requests)
client_http2 = PerformanceClient(base_url, api_key, http_version=2)
```

```javascript
// HTTP/1.1 (default)
const client1 = new PerformanceClient(baseUrl, apiKey, 1);

// HTTP/2
const client2 = new PerformanceClient(baseUrl, apiKey, 2);
```

#### Connection Pooling
Share connection pools across multiple client instances:

```python
from baseten_performance_client import HttpClientWrapper

# Create shared wrapper
wrapper = HttpClientWrapper(http_version=1)

# Reuse across multiple clients
client1 = PerformanceClient(base_url="https://api1.example.com", client_wrapper=wrapper)
client2 = PerformanceClient(base_url="https://api2.example.com", client_wrapper=wrapper)
```

```javascript
const { HttpClientWrapper } = require('baseten-performance-client');

// Create shared wrapper
const wrapper = new HttpClientWrapper(1);

// Reuse across multiple clients
const client1 = new PerformanceClient(baseUrl1, apiKey, 1, wrapper);
const client2 = new PerformanceClient(baseUrl2, apiKey, 1, wrapper);
```

#### Custom Headers
Add custom headers to batch requests:

```python
response = client.batch_post(
    url_path="/v1/embeddings",
    payloads=payloads,
    custom_headers={
        "x-custom-header": "value",
        "authorization": "Bearer token"
    }
)
```

```javascript
const { RequestProcessingPreference } = require('baseten-performance-client');

const preference = new RequestProcessingPreference(32, undefined, undefined, 360.0, 0.5, 360.0);
const response = await client.batchPost(
    "/v1/embeddings",
    payloads,
    {"x-custom-header": "value"}, // custom headers
    preference // preference parameter
);
```

#### Cancellation Support
The client supports cancellation of long-running operations through `CancellationToken`. This allows you to cancel batch operations that are in progress.

```python
from baseten_performance_client import CancellationToken, RequestProcessingPreference

# Create a cancellation token
cancel_token = CancellationToken()

# Configure preference with cancellation token
preference = RequestProcessingPreference(
    max_concurrent_requests=32,
    batch_size=16,
    timeout_s=360.0,
    cancel_token=cancel_token
)

# Start a long-running operation (in a separate thread or async context)
import threading
import time

def long_operation():
    try:
        response = client.embed(
            input=["large batch of texts"] * 1000,
            model="embedding-model",
            preference=preference
        )
        print("Operation completed successfully")
    except ValueError as e:
        if "cancelled" in str(e):
            print("Operation was cancelled")
        else:
            print(f"Error: {e}")

# Start the operation
operation_thread = threading.Thread(target=long_operation)
operation_thread.start()

# Cancel after 2 seconds
time.sleep(2)
cancel_token.cancel()
print("Cancellation requested")

operation_thread.join()
```

```javascript
const { CancellationToken, RequestProcessingPreference } = require('baseten-performance-client');

// Create a cancellation token
const cancelToken = new CancellationToken();

// Configure preference with cancellation token
const preference = new RequestProcessingPreference(
    32,        // maxConcurrentRequests
    16,        // batchSize
    undefined, // maxCharsPerRequest
    360.0,     // timeoutS
    undefined, // hedgeDelay
    undefined, // totalTimeoutS
    undefined, // hedgeBudgetPct
    undefined, // retryBudgetPct
    undefined, // maxRetries
    undefined, // initialBackoffMs
    cancelToken  // cancellation token
);

// Start a long-running operation
const operation = client.embed(
    ["large batch of texts"].concat(Array(1000).fill("sample text")),
    "embedding-model",
    undefined, undefined, undefined, // encodingFormat, dimensions, user
    preference
);

// Cancel after 2 seconds
setTimeout(() => {
    cancelToken.cancel();
    console.log("Cancellation requested");
}, 2000);

try {
    const response = await operation;
    console.log("Operation completed successfully");
} catch (error) {
    if (error.message.includes("cancelled")) {
        console.log("Operation was cancelled");
    } else {
        console.log(`Error: ${error.message}`);
    }
}
```

**Key Features:**
- **Immediate Cancellation**: When `cancel()` is called, all in-flight requests are aborted
- **Resource Cleanup**: Cancellation triggers automatic cleanup of spawned tasks and connections
- **Ctrl+C Support**: Python operations automatically respond to Ctrl+C interrupts
- **Token Sharing**: The same token can be used across multiple operations for coordinated cancellation
- **Status Checking**: Use `is_cancelled()` to check if cancellation has been requested

**Use Cases:**
- Timeout-based cancellation for long-running batch operations
- User-initiated cancellation in interactive applications
- Coordinated cancellation across multiple concurrent operations
- Graceful shutdown in server applications

## Rust

```rust
use baseten_performance_client_core::{PerformanceClientCore, ClientError};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("BASETEN_API_KEY").expect("BASETEN_API_KEY not set");
    let base_url = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";

    let client = PerformanceClientCore::new(base_url, Some(api_key), None, None); // http_version, client_wrapper

    // Embedding example
    let texts = vec!["Hello world".to_string(), "Example text".to_string()];
    let embedding_response = client.embed(
        texts,
        "my_model".to_string(),
        Some(16),                   // batch_size
        Some(32),                   // max_concurrent_requests
        Some(360.0),                // timeout_s
        Some(256000),               // max_chars_per_request
        Some(0.5),                  // hedge_delay
        Some(360.0),                // total_timeout_s
    ).await?;

    println!("Model: {}", embedding_response.model);
    println!("Total tokens: {}", embedding_response.usage.total_tokens);

    // Batch POST example
    let payloads = vec![
        serde_json::json!({"model": "my_model", "input": ["Rust sample 1"]}),
        serde_json::json!({"model": "my_model", "input": ["Rust sample 2"]}),
    ];

    let batch_response = client.batch_post(
        "/v1/embeddings".to_string(),
        payloads,
        Some(32),                   // max_concurrent_requests
        Some(360.0),                // timeout_s
        Some(0.5),                  // hedge_delay
        Some(360.0),                // total_timeout_s
        None,                       // custom_headers
    ).await?;

    println!("Batch POST total time: {:.4}s", batch_response.total_time);

    Ok(())
}
```

## Configuration

### Environment Variables

- `BASETEN_API_KEY`: Your Baseten API key (also checks `OPENAI_API_KEY` as fallback)
- `PERFORMANCE_CLIENT_LOG_LEVEL`: Set the logging level for the performance client (overrides `RUST_LOG`)
  - Valid values: `trace`, `debug`, `info`, `warn`, `error`
  - Default: `warn`
  - Priority: `PERFORMANCE_CLIENT_LOG_LEVEL` > `RUST_LOG` > default
- `PERFORMANCE_CLIENT_REQUEST_ID_PREFIX`: Custom prefix for request IDs (default: "perfclient")

### Logging Examples

```bash
# Use info level logging
PERFORMANCE_CLIENT_LOG_LEVEL=info python your_script.py

# Use debug level logging
PERFORMANCE_CLIENT_LOG_LEVEL=info cargo run

# Traditional RUST_LOG still works (lower priority)
RUST_LOG=debug python your_script.py

# PERFORMANCE_CLIENT_LOG_LEVEL takes precedence
PERFORMANCE_CLIENT_LOG_LEVEL=error RUST_LOG=trace python your_script.py  # Uses error level
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
