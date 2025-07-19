# High performance client for Baseten.co - Node.js Bindings

This library provides a high-performance Node.js client for Baseten.co endpoints including embeddings, reranking, and classification. It was built for massive concurrent POST requests to any URL, also outside of baseten.co. The PerformanceClient is built on top of Rust (using napi-rs), reqwest and tokio and is MIT licensed.

Similar to the Python version, this client supports >1200 rps per client and was benchmarked in [our blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/).

![benchmarks](https://www.baseten.co/_next/image/?url=https%3A%2F%2Fwww.datocms-assets.com%2F104802%2F1749832130-diagram-9.png%3Fauto%3Dformat%26fit%3Dmax%26w%3D1200&w=3840&q=75)

## Installation

```bash
npm install @basetenlabs/performance-client
```

## Usage

### Basic Setup

Since different endpoints require different clients, you'll typically need to create separate clients for embeddings and reranking deployments.

```javascript
const { PerformanceClient } = require('@basetenlabs/performance-client');

const apiKey = process.env.BASETEN_API_KEY;
const embedBaseUrl = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
const rerankBaseUrl = "https://model-abc123.api.baseten.co/environments/production/sync";

// Create separate clients for different endpoints
const embedClient = new PerformanceClient(embedBaseUrl, apiKey);
const rerankClient = new PerformanceClient(rerankBaseUrl, apiKey);
```

### Embeddings

```javascript
const texts = ["Hello world", "Example text", "Another sample"];

try {
    const response = embedClient.embed(
        texts,
        "text-embedding-3-small", // model
        null, // encoding_format
        null, // dimensions
        null, // user
        8,    // max_concurrent_requests
        2,    // batch_size
        30    // timeout_s
    );

    console.log(`Model used: ${response.model}`);
    console.log(`Total tokens used: ${response.usage.total_tokens}`);
    console.log(`Total time: ${response.total_time.toFixed(4)}s`);

    if (response.individual_request_times) {
        response.individual_request_times.forEach((time, i) => {
            console.log(`  Time for batch ${i}: ${time.toFixed(4)}s`);
        });
    }

    response.data.forEach((embedding, i) => {
        console.log(`Embedding for text ${i} (original input index ${embedding.index}):`);
        console.log(`  First 3 dimensions: ${embedding.embedding.slice(0, 3)}`);
        console.log(`  Length: ${embedding.embedding.length}`);
    });
} catch (error) {
    console.error('Embedding failed:', error.message);
}
```

### Reranking

```javascript
const query = "What is the best framework?";
const documents = [
    "Machine learning is a subset of artificial intelligence",
    "JavaScript is a programming language",
    "Deep learning uses neural networks",
    "Python is popular for data science"
];

try {
    const response = rerankClient.rerank(
        query,
        documents,
        false, // raw_scores
        true,  // return_text
        false, // truncate
        "Right", // truncation_direction
        4,     // max_concurrent_requests
        2,     // batch_size
        30     // timeout_s
    );

    console.log(`Reranked ${response.data.length} documents`);
    console.log(`Total time: ${response.total_time.toFixed(4)}s`);

    response.data.forEach((result, i) => {
        console.log(`${i + 1}. Score: ${result.score.toFixed(3)} - ${result.text?.substring(0, 50)}...`);
    });
} catch (error) {
    console.error('Reranking failed:', error.message);
}
```

### Classification

```javascript
const textsToClassify = [
    "This is great!",
    "I did not like it.",
    "Neutral experience."
];

try {
    const response = rerankClient.classify(
        textsToClassify,
        false, // raw_scores
        false, // truncate
        "Right", // truncation_direction
        4,     // max_concurrent_requests
        2,     // batch_size
        30     // timeout_s
    );

    console.log(`Classified ${response.data.length} texts`);
    console.log(`Total time: ${response.total_time.toFixed(4)}s`);

    response.data.forEach((group, i) => {
        console.log(`Text ${i + 1}:`);
        group.forEach(result => {
            console.log(`  ${result.label}: ${result.score.toFixed(3)}`);
        });
    });
} catch (error) {
    console.error('Classification failed:', error.message);
}
```

### General Batch POST

The batch_post method is generic and can be used to send POST requests to any URL, not limited to Baseten endpoints:

```javascript
const payloads = [
    { "model": "text-embedding-3-small", "input": ["Hello"] },
    { "model": "text-embedding-3-small", "input": ["World"] }
];

try {
    const response = embedClient.batchPost(
        "/v1/embeddings", // URL path
        payloads,
        4,  // max_concurrent_requests
        30  // timeout_s
    );

    console.log(`Processed ${response.data.length} batch requests`);
    console.log(`Total time: ${response.total_time.toFixed(4)}s`);

    response.data.forEach((result, i) => {
        console.log(`Request ${i + 1}: ${JSON.stringify(result).substring(0, 100)}...`);
    });

    // Access response headers and individual request times
    response.response_headers.forEach((headers, i) => {
        console.log(`Response ${i + 1} headers:`, headers);
    });

    response.individual_request_times.forEach((time, i) => {
        console.log(`Request ${i + 1} took: ${time.toFixed(4)}s`);
    });
} catch (error) {
    console.error('Batch POST failed:', error.message);
}
```

## API Reference

### Constructor

```javascript
new PerformanceClient(baseUrl, apiKey)
```

- `baseUrl` (string): The base URL for the API endpoint
- `apiKey` (string, optional): API key. If not provided, will use `BASETEN_API_KEY` or `OPENAI_API_KEY` environment variables

### Methods

#### embed(input, model, encoding_format, dimensions, user, max_concurrent_requests, batch_size, timeout_s)

- `input` (Array<string>): List of texts to embed
- `model` (string): Model name
- `encoding_format` (string, optional): Encoding format
- `dimensions` (number, optional): Number of dimensions
- `user` (string, optional): User identifier
- `max_concurrent_requests` (number, optional): Maximum concurrent requests (default: 32)
- `batch_size` (number, optional): Batch size (default: 128)
- `timeout_s` (number, optional): Timeout in seconds (default: 3600)

#### rerank(query, texts, raw_scores, return_text, truncate, truncation_direction, max_concurrent_requests, batch_size, timeout_s)

- `query` (string): Query text
- `texts` (Array<string>): List of texts to rerank
- `raw_scores` (boolean, optional): Return raw scores (default: false)
- `return_text` (boolean, optional): Return text in response (default: false)
- `truncate` (boolean, optional): Truncate long texts (default: false)
- `truncation_direction` (string, optional): "Left" or "Right" (default: "Right")
- `max_concurrent_requests` (number, optional): Maximum concurrent requests (default: 32)
- `batch_size` (number, optional): Batch size (default: 128)
- `timeout_s` (number, optional): Timeout in seconds (default: 3600)

#### classify(inputs, raw_scores, truncate, truncation_direction, max_concurrent_requests, batch_size, timeout_s)

- `inputs` (Array<string>): List of texts to classify
- `raw_scores` (boolean, optional): Return raw scores (default: false)
- `truncate` (boolean, optional): Truncate long texts (default: false)
- `truncation_direction` (string, optional): "Left" or "Right" (default: "Right")
- `max_concurrent_requests` (number, optional): Maximum concurrent requests (default: 32)
- `batch_size` (number, optional): Batch size (default: 128)
- `timeout_s` (number, optional): Timeout in seconds (default: 3600)

#### batchPost(url_path, payloads, max_concurrent_requests, timeout_s)

- `url_path` (string): URL path for the POST request
- `payloads` (Array<Object>): List of JSON payloads
- `max_concurrent_requests` (number, optional): Maximum concurrent requests (default: 32)
- `timeout_s` (number, optional): Timeout in seconds (default: 3600)

## Error Handling

The client throws standard JavaScript errors for various failure cases:

```javascript
try {
    const response = embedClient.embed(texts, "model");
} catch (error) {
    if (error.message.includes('cannot be empty')) {
        console.error('Parameter validation error:', error.message);
    } else if (error.message.includes('HTTP')) {
        console.error('Network error:', error.message);
    } else {
        console.error('Other error:', error.message);
    }
}
```

## Testing

Run the test suite:

```bash
npm test
```

The tests use a simple built-in test framework and validate parameter handling, constructor behavior, and error conditions.

## Development

To build the native module:

```bash
# Install dependencies
npm install

# Build release version
npm run build

# Build debug version
npm run build:debug
```

## Benchmarks

Like the Python version, this Node.js client provides significant performance improvements over standard HTTP clients, especially for high-throughput embedding and reranking workloads.

## License

MIT License

## Acknowledgements:
Venkatesh Narayan (Clay.com) for the prototpe of this here https://github.com/basetenlabs/truss/pull/1778
and Suren (Baseten) for getting a PoC and protyping the release pipeline. https://github.com/suren-atoyan/rust-ts-package
