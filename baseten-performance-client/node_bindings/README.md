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
const { PerformanceClient, RequestProcessingPreference } = require('@basetenlabs/performance-client');

const apiKey = process.env.BASETEN_API_KEY;
const embedBaseUrl = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
const rerankBaseUrl = "https://model-abc123.api.baseten.co/environments/production/sync";

// Create separate clients for different endpoints
const embedClient = new PerformanceClient(embedBaseUrl, apiKey);
const rerankClient = new PerformanceClient(rerankBaseUrl, apiKey);
```

## RequestProcessingPreference

All API methods accept an optional `RequestProcessingPreference` object that configures request processing behavior:

```javascript
const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

// Create with defaults
const pref = new RequestProcessingPreference();

// Or customize all parameters
const pref = new RequestProcessingPreference(
    64,     // maxConcurrentRequests (default: 128)
    32,     // batchSize (default: 128)
    30.0,   // timeoutS (default: 3600)
    10000,  // maxCharsPerRequest (optional)
    0.5,    // hedgeDelay (optional)
    600.0,  // totalTimeoutS (optional)
    0.10,   // hedgeBudgetPct (default: 0.10)
    0.05    // retryBudgetPct (default: 0.05)
);

// Access properties
console.log(pref.maxConcurrentRequests);  // 64
console.log(pref.batchSize);              // 32
console.log(pref.timeoutS);               // 30.0
console.log(pref.hedgeBudgetPct);         // 0.10
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxConcurrentRequests` | 128 | Maximum number of concurrent HTTP requests |
| `batchSize` | 128 | Number of items per batch/request |
| `timeoutS` | 3600.0 | Timeout for individual requests in seconds |
| `maxCharsPerRequest` | null | Character-based batching limit (50-256,000) |
| `hedgeDelay` | null | Request hedging delay in seconds (min 0.2s) |
| `totalTimeoutS` | null | Total timeout for all batched requests |
| `hedgeBudgetPct` | 0.10 | Percentage of total requests that can be hedged |
| `retryBudgetPct` | 0.05 | Percentage of total requests that can be retried |

### Embeddings

```javascript
const texts = ["Hello world", "Example text", "Another sample"];

// Simple usage with defaults
const response = await embedClient.embed(texts, "text-embedding-3-small");

// With custom preferences
const pref = new RequestProcessingPreference(8, 2, 30);
const response = await embedClient.embed(
    texts,
    "text-embedding-3-small",  // model
    null,                       // encoding_format
    null,                       // dimensions
    null,                       // user
    pref                        // preference
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

const pref = new RequestProcessingPreference(4, 2, 30);
const response = await rerankClient.rerank(
    query,
    documents,
    false,   // raw_scores
    null,    // model
    true,    // return_text
    false,   // truncate
    "Right", // truncation_direction
    pref     // preference
);

console.log(`Reranked ${response.data.length} documents`);
console.log(`Total time: ${response.total_time.toFixed(4)}s`);

response.data.forEach((result, i) => {
    console.log(`${i + 1}. Score: ${result.score.toFixed(3)} - ${result.text?.substring(0, 50)}...`);
});
```

### Classification

```javascript
const textsToClassify = [
    "This is great!",
    "I did not like it.",
    "Neutral experience."
];

const pref = new RequestProcessingPreference(4, 2, 30);
const response = await rerankClient.classify(
    textsToClassify,
    null,    // model
    false,   // raw_scores
    false,   // truncate
    "Right", // truncation_direction
    pref     // preference
);

console.log(`Classified ${response.data.length} texts`);
console.log(`Total time: ${response.total_time.toFixed(4)}s`);

response.data.forEach((group, i) => {
    console.log(`Text ${i + 1}:`);
    group.forEach(result => {
        console.log(`  ${result.label}: ${result.score.toFixed(3)}`);
    });
});
```

### General Batch POST

The batchPost method is generic and can be used to send POST requests to any URL, not limited to Baseten endpoints:

```javascript
const payloads = [
    { "model": "text-embedding-3-small", "input": ["Hello"] },
    { "model": "text-embedding-3-small", "input": ["World"] }
];

const pref = new RequestProcessingPreference(4, null, 30);
const response = await embedClient.batchPost(
    "/v1/embeddings",  // URL path
    payloads,
    pref               // preference
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
```

## API Reference

### Constructor

```javascript
new PerformanceClient(baseUrl, apiKey)
```

- `baseUrl` (string): The base URL for the API endpoint
- `apiKey` (string, optional): API key. If not provided, will use `BASETEN_API_KEY` or `OPENAI_API_KEY` environment variables

### RequestProcessingPreference Constructor

```javascript
new RequestProcessingPreference(
    maxConcurrentRequests,  // number, optional (default: 128)
    batchSize,              // number, optional (default: 128)
    timeoutS,               // number, optional (default: 3600)
    maxCharsPerRequest,     // number, optional (default: null)
    hedgeDelay,             // number, optional (default: null)
    totalTimeoutS,          // number, optional (default: null)
    hedgeBudgetPct,         // number, optional (default: 0.10)
    retryBudgetPct          // number, optional (default: 0.05)
)
```

### Methods

#### embed(input, model, encodingFormat, dimensions, user, preference)

- `input` (Array<string>): List of texts to embed
- `model` (string): Model name
- `encodingFormat` (string, optional): Encoding format
- `dimensions` (number, optional): Number of dimensions
- `user` (string, optional): User identifier
- `preference` (RequestProcessingPreference, optional): Processing configuration

#### rerank(query, texts, rawScores, model, returnText, truncate, truncationDirection, preference)

- `query` (string): Query text
- `texts` (Array<string>): List of texts to rerank
- `rawScores` (boolean, optional): Return raw scores (default: false)
- `model` (string, optional): Model name
- `returnText` (boolean, optional): Return text in response (default: false)
- `truncate` (boolean, optional): Truncate long texts (default: false)
- `truncationDirection` (string, optional): "Left" or "Right" (default: "Right")
- `preference` (RequestProcessingPreference, optional): Processing configuration

#### classify(inputs, model, rawScores, truncate, truncationDirection, preference)

- `inputs` (Array<string>): List of texts to classify
- `model` (string, optional): Model name
- `rawScores` (boolean, optional): Return raw scores (default: false)
- `truncate` (boolean, optional): Truncate long texts (default: false)
- `truncationDirection` (string, optional): "Left" or "Right" (default: "Right")
- `preference` (RequestProcessingPreference, optional): Processing configuration

#### batchPost(urlPath, payloads, preference, customHeaders)

- `urlPath` (string): URL path for the POST request
- `payloads` (Array<Object>): List of JSON payloads
- `preference` (RequestProcessingPreference, optional): Processing configuration
- `customHeaders` (Object, optional): Custom headers to include

## Error Handling

The client throws standard JavaScript errors for various failure cases:

```javascript
try {
    const pref = new RequestProcessingPreference(8, 2, 30);
    const response = await embedClient.embed(texts, "model", null, null, null, pref);
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
Venkatesh Narayan (Clay.com) for the prototype of this here https://github.com/basetenlabs/truss/pull/1778
and Suren (Baseten) for getting a PoC and prototyping the release pipeline. https://github.com/suren-atoyan/rust-ts-package
