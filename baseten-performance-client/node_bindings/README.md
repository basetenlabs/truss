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
const { PerformanceClient, HttpClientWrapper } = require('@basetenlabs/performance-client');

const apiKey = process.env.BASETEN_API_KEY;
const embedBaseUrl = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
const rerankBaseUrl = "https://model-abc123.api.baseten.co/environments/production/sync";

// Create separate clients for different endpoints
const embedClient = new PerformanceClient(embedBaseUrl, apiKey);
const rerankClient = new PerformanceClient(rerankBaseUrl, apiKey);

// Advanced setup with custom HTTP version and client wrapper
const httpWrapper = new HttpClientWrapper(2); // Use HTTP/2
const advancedClient = new PerformanceClient(baseUrl, apiKey, 2, httpWrapper);
```

### Embeddings

```javascript
const texts = ["Hello world", "Example text", "Another sample"];

const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

try {
    const preference = new RequestProcessingPreference(
        8,    // max_concurrent_requests
        2,    // batch_size
        undefined, // max_chars_per_request
        30,   // timeout_s
        undefined, // hedge_delay
        undefined, // total_timeout_s
        undefined, // hedge_budget_pct
        undefined, // retry_budget_pct
        undefined, // max_retries
        undefined  // initial_backoff_ms
    );
    const response = await embedClient.embed(
        texts,
        "text-embedding-3-small", // model
        null, null, null, // encoding_format, dimensions, user
        preference // preference parameter
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

const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

try {
    const preference = new RequestProcessingPreference(
        4,     // max_concurrent_requests
        2,     // batch_size
        undefined, // max_chars_per_request
        30,    // timeout_s
        undefined, // hedge_delay
        undefined, // total_timeout_s
        undefined, // hedge_budget_pct
        undefined, // retry_budget_pct
        undefined, // max_retries
        undefined  // initial_backoff_ms
    );
    const response = await rerankClient.rerank(
        query,
        documents,
        false, // raw_scores
        true,  // return_text
        false, // truncate
        "Right", // truncation_direction
        preference // preference parameter
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

const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

try {
    const preference = new RequestProcessingPreference(
        4,     // max_concurrent_requests
        2,     // batch_size
        undefined, // max_chars_per_request
        30,    // timeout_s
        undefined, // hedge_delay
        undefined, // total_timeout_s
        undefined, // hedge_budget_pct
        undefined, // retry_budget_pct
        undefined, // max_retries
        undefined  // initial_backoff_ms
    );
    const response = await rerankClient.classify(
        textsToClassify,
        false, // raw_scores
        false, // truncate
        "Right", // truncation_direction
        preference // preference parameter
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

const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

try {
    const preference = new RequestProcessingPreference(
        4,  // max_concurrent_requests
        undefined, // batch_size
        undefined, // max_chars_per_request
        30, // timeout_s
        undefined, // hedge_delay
        undefined, // total_timeout_s
        undefined, // hedge_budget_pct
        undefined, // retry_budget_pct
        undefined, // max_retries
        undefined  // initial_backoff_ms
    );
    const response = await embedClient.batchPost(
        "/v1/embeddings", // URL path
        payloads,
        undefined, // custom headers
        preference // preference parameter
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

### Advanced Features

#### RequestProcessingPreference

The `RequestProcessingPreference` class provides a unified way to configure all request processing parameters. This is the recommended approach for advanced configuration as it provides better type safety and clearer intent.

```javascript
const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

// Create a preference with custom settings
const preference = new RequestProcessingPreference(
    64,        // maxConcurrentRequests (default: 128)
    32,        // batchSize (default: 128)
    undefined, // maxCharsPerRequest
    30.0,      // timeoutS (default: 3600.0)
    0.5,       // hedgeDelay
    undefined, // totalTimeoutS
    0.15,      // hedgeBudgetPct (default: 0.10)
    0.08,      // retryBudgetPct (default: 0.05)
    3,         // maxRetries (default: 4)
    250        // initialBackoffMs (default: 125)
);

// Use with any method
const response = await embedClient.embed(
    ["text1", "text2"],
    "my_model",
    undefined, undefined, undefined, // encodingFormat, dimensions, user
    preference // preference parameter
);
```

**Budget Percentages:**
- `hedgeBudgetPct`: Percentage of total requests allocated for hedging (default: 10%)
- `retryBudgetPct`: Percentage of total requests allocated for retries (default: 5%)
- Maximum allowed: 300% for both budgets

**Retry Configuration:**
- `maxRetries`: Maximum number of HTTP retries (default: 4, max: 4)
- `initialBackoffMs`: Initial backoff duration in milliseconds (default: 125, range: 50-30000)
- Backoff uses exponential backoff with jitter

#### Request Hedging
The client supports request hedging for improved latency:

```javascript
const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

const preference = new RequestProcessingPreference(
    8, 2, 100000, 30, 0.5, 60, 0.1, 0.05, 3, 250  // maxConcurrentRequests, batchSize, maxCharsPerRequest, timeoutS, hedgeDelay, totalTimeoutS, hedgeBudgetPct, retryBudgetPct, maxRetries, initialBackoffMs
);
const response = await embedClient.embed(
    texts,
    "text-embedding-3-small",
    null, null, null, // encoding_format, dimensions, user
    preference // preference parameter
);
```

#### Retry Configuration
Configure retry behavior and backoff settings:

```javascript
const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

// Configure for more aggressive retrying
const preference = new RequestProcessingPreference(
    32,        // maxConcurrentRequests
    16,        // batchSize
    undefined, // maxCharsPerRequest
    60.0,      // timeoutS
    undefined, // hedgeDelay
    undefined, // totalTimeoutS
    undefined, // hedgeBudgetPct
    0.10,      // retryBudgetPct (10% for retries)
    4,         // maxRetries (maximum allowed)
    500        // initialBackoffMs (start with 500ms backoff)
);

const response = await embedClient.embed(
    texts,
    "text-embedding-3-small",
    null, null, null, // encoding_format, dimensions, user
    preference // preference parameter
);
```

#### Custom Headers
Use custom headers with batchPost:

```javascript
const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

const preference = new RequestProcessingPreference(4, undefined, undefined, 30, undefined, undefined, undefined, undefined, undefined, undefined);
const response = await client.batchPost(
    "/v1/embeddings",
    payloads,
    { "x-custom-header": "value" }, // custom headers
    preference // preference parameter
);
```

#### HTTP Version Selection
Choose between HTTP/1.1 and HTTP/2:

```javascript
// HTTP/1.1 (default for compatibility)
const clientHttp1 = new PerformanceClient(baseUrl, apiKey, 1);

// HTTP/2 (better performance for multiple requests)
const clientHttp2 = new PerformanceClient(baseUrl, apiKey, 2);
```

## API Reference

### Constructors

#### PerformanceClient

```javascript
new PerformanceClient(baseUrl, apiKey?, httpVersion?, clientWrapper?)
```

- `baseUrl` (string): The base URL for the API endpoint
- `apiKey` (string, optional): API key. If not provided, will use `BASETEN_API_KEY` or `OPENAI_API_KEY` environment variables
- `httpVersion` (number, optional): HTTP version to use (1 for HTTP/1.1, 2 for HTTP/2). Default: 2
- `clientWrapper` (HttpClientWrapper, optional): Custom HTTP client wrapper for advanced configuration

#### RequestProcessingPreference

```javascript
new RequestProcessingPreference(maxConcurrentRequests?, batchSize?, maxCharsPerRequest?, timeoutS?, hedgeDelay?, totalTimeoutS?, hedgeBudgetPct?, retryBudgetPct?, maxRetries?, initialBackoffMs?)
```

- `maxConcurrentRequests` (number, optional): Maximum number of parallel requests (default: 128)
- `batchSize` (number, optional): Number of items per batch (default: 128)
- `maxCharsPerRequest` (number, optional): Character-based batching limit (default: undefined)
- `timeoutS` (number, optional): Per-request timeout in seconds (default: 3600.0)
- `hedgeDelay` (number, optional): Request hedging delay in seconds (default: undefined)
- `totalTimeoutS` (number, optional): Total timeout for the entire operation in seconds (default: undefined)
- `hedgeBudgetPct` (number, optional): Hedge budget percentage (default: 0.10, range: 0.0-3.0)
- `retryBudgetPct` (number, optional): Retry budget percentage (default: 0.05, range: 0.0-3.0)
- `maxRetries` (number, optional): Maximum number of HTTP retries (default: 4, max: 4)
- `initialBackoffMs` (number, optional): Initial backoff duration in milliseconds (default: 125, range: 50-30000)

### Methods

#### embed(input, model, encodingFormat?, dimensions?, user?, preference?)

- `input` (Array<string>): List of texts to embed
- `model` (string): Model name
- `encodingFormat` (string, optional): Encoding format
- `dimensions` (number, optional): Number of dimensions
- `user` (string, optional): User identifier
- `preference` (RequestProcessingPreference, optional): Advanced configuration preference object

#### rerank(query, texts, rawScores?, model?, returnText?, truncate?, truncationDirection?, preference?)

- `query` (string): Query text
- `texts` (Array<string>): List of texts to rerank
- `rawScores` (boolean, optional): Return raw scores (default: false)
- `model` (string, optional): Model name for reranking
- `returnText` (boolean, optional): Return text in response (default: false)
- `truncate` (boolean, optional): Truncate long texts (default: false)
- `truncationDirection` (string, optional): "Left" or "Right" (default: "Right")
- `preference` (RequestProcessingPreference, optional): Advanced configuration preference object

#### classify(inputs, rawScores?, model?, truncate?, truncationDirection?, preference?)

- `inputs` (Array<string>): List of texts to classify
- `rawScores` (boolean, optional): Return raw scores (default: false)
- `model` (string, optional): Model name for classification
- `truncate` (boolean, optional): Truncate long texts (default: false)
- `truncationDirection` (string, optional): "Left" or "Right" (default: "Right")
- `preference` (RequestProcessingPreference, optional): Advanced configuration preference object

#### batchPost(urlPath, payloads, customHeaders?, preference?)

- `urlPath` (string): URL path for the POST request
- `payloads` (Array<Object>): List of JSON payloads
- `customHeaders` (Record<string, string>, optional): Custom headers to include with each request
- `preference` (RequestProcessingPreference, optional): Advanced configuration preference object

## Error Handling

The client throws standard JavaScript errors for various failure cases:

```javascript
const { RequestProcessingPreference } = require('@basetenlabs/performance-client');

try {
    const preference = new RequestProcessingPreference();
    const response = await embedClient.embed(texts, "model", null, null, null, preference);
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

## Releasing

To release a new version of the Node.js bindings:

1. **Update the version in `Cargo.toml`** - This is the source of truth for versioning
2. **Sync versions with NAPI** - Run the version sync command to update `package.json` and regenerate code:
   ```bash
   napi version
   ```
3. **Build the project** - This regenerates the `index.js` file with the correct version checks:
   ```bash
   npm run build
   ```
4. **Commit the changes** - Include both `Cargo.toml` and `package.json` updates:
   ```bash
   git add Cargo.toml package.json
   git commit -m "chore: bump version to x.y.z"
   ```
5. **Publish** - The CI will automatically publish when run via workflow dispatch and setting "release" or "next" as the publish type

### Important Notes
- Always update `Cargo.toml` first, then run `napi version` to sync to `package.json`
- The `napi version` command ensures version consistency between Rust and Node.js
- Rebuilding after version sync is crucial to update hardcoded version checks in the generated `index.js` file
- The CI will fail if `package.json` version doesn't match the built-in version checks

## Benchmarks

Like the Python version, this Node.js client provides significant performance improvements over standard HTTP clients, especially for high-throughput embedding and reranking workloads.

## License

MIT License

## Acknowledgements:
Venkatesh Narayan (Clay.com) for the prototpe of this here https://github.com/basetenlabs/truss/pull/1778
and Suren (Baseten) for getting a PoC and protyping the release pipeline. https://github.com/suren-atoyan/rust-ts-package
