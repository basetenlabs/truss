# High Performance Node.js Client for Baseten.co

This library provides a high-performance Node.js client for Baseten.co endpoints including embeddings, reranking, and classification. It was built for massive concurrent POST requests to any URL, also outside of baseten.co. The client is built on top of Rust using napi-rs, reqwest, and tokio for maximum performance while providing a simple JavaScript API.

## Features

- **High Performance**: Built with Rust and tokio for maximum concurrent throughput
- **Concurrent Requests**: Handles thousands of concurrent requests efficiently
- **Multiple AI APIs**: Works with Baseten.co, OpenAI, Mixedbread, and other compatible APIs
- **Batch Processing**: Intelligent batching for optimal performance
- **TypeScript Support**: Full TypeScript type definitions included
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Installation

```bash
npm install @baseten/performance-client
```

## Usage

```javascript
const { PerformanceClient } = require('@baseten/performance-client');

const apiKey = process.env.BASETEN_API_KEY;
const baseUrl = "https://model-yqv0rjjw.api.baseten.co/environments/production/sync";
// Also works with OpenAI or Mixedbread APIs
// const baseUrl = "https://api.openai.com" or "https://api.mixedbread.com"

const client = new PerformanceClient(baseUrl, apiKey);
```

### Embeddings

#### Basic Embedding

```javascript
const texts = ["Hello world", "Example text", "Another sample"];
const response = client.embed(
    texts,
    "my_model",
    null, // encoding_format
    null, // dimensions
    null, // user
    32,   // max_concurrent_requests
    4,    // batch_size
    360   // timeout_s
);

// Accessing embedding data
console.log(`Model used: ${response.model}`);
console.log(`Total tokens used: ${response.usage.total_tokens}`);
console.log(`Total time: ${response.total_time.toFixed(4)}s`);

if (response.individual_request_times) {
    response.individual_request_times.forEach((time, i) => {
        console.log(`Time for batch ${i}: ${time.toFixed(4)}s`);
    });
}

response.data.forEach((embeddingData, i) => {
    console.log(`Embedding for text ${i} (original index ${embeddingData.index}):`);
    console.log(`  First 3 dimensions: ${embeddingData.embedding.slice(0, 3)}`);
    console.log(`  Length: ${embeddingData.embedding.length}`);
});
```

#### With TypeScript

```typescript
import { PerformanceClient } from '@baseten/performance-client';

const client = new PerformanceClient(baseUrl, apiKey);

const texts: string[] = ["Hello world", "Example text"];
const response = client.embed(
    texts,
    "my_model",
    null, // encoding_format
    null, // dimensions
    null, // user
    32,   // max_concurrent_requests
    4,    // batch_size
    360   // timeout_s
);

// TypeScript will provide full type safety
console.log(`Model: ${response.model}`);
response.data.forEach(item => {
    console.log(`Index: ${item.index}, Embedding length: ${item.embedding.length}`);
});
```

### Reranking

```javascript
const query = "What is the best framework?";
const documents = ["Doc 1 text", "Doc 2 text", "Doc 3 text"];

const rerankResponse = client.rerank(
    query,
    documents,
    false, // raw_scores
    true,  // return_text
    false, // truncate
    "Right", // truncation_direction
    16,    // max_concurrent_requests
    2,     // batch_size
    360    // timeout_s
);

rerankResponse.data.forEach(result => {
    console.log(`Index: ${result.index}, Score: ${result.score}`);
    if (result.text) {
        console.log(`Text: ${result.text}`);
    }
});
```

### Classification

```javascript
const textsToClassify = [
    "This is great!",
    "I did not like it.",
    "Neutral experience."
];

const classifyResponse = client.classify(
    textsToClassify,
    false, // raw_scores
    false, // truncate
    "Right", // truncation_direction
    16,    // max_concurrent_requests
    2,     // batch_size
    360    // timeout_s
);

classifyResponse.data.forEach(group => {
    group.forEach(result => {
        console.log(`Label: ${result.label}, Score: ${result.score}`);
    });
});
```

### General Batch POST

The batch_post method is generic and can be used to send POST requests to any URL, not limited to Baseten endpoints. The input and output can be any JSON data.

```javascript
const payload1 = { model: "my_model", input: ["Batch request sample 1"] };
const payload2 = { model: "my_model", input: ["Batch request sample 2"] };

const batchResponse = client.batch_post(
    "/v1/embeddings", // URL path
    [payload1, payload2], // payloads
    96,  // max_concurrent_requests
    360  // timeout_s
);

console.log(`Total time for batch POST: ${batchResponse.total_time.toFixed(4)}s`);

batchResponse.data.forEach((respData, i) => {
    console.log(`Response ${i + 1}:`);
    console.log(`  Data:`, respData);
    console.log(`  Headers:`, batchResponse.response_headers[i]);
    console.log(`  Time taken: ${batchResponse.individual_request_times[i].toFixed(4)}s`);
});
```

## API Reference

### PerformanceClient

#### Constructor

```javascript
new PerformanceClient(baseUrl, apiKey?)
```

- `baseUrl`: Base URL for the API
- `apiKey`: Optional API key (can also be set via `BASETEN_API_KEY` or `OPENAI_API_KEY` environment variables)

#### Methods

##### `embed(input, model, encodingFormat?, dimensions?, user?, maxConcurrentRequests?, batchSize?, timeoutS?)`

Process embeddings requests.

- `input`: Array of strings to embed
- `model`: Model name
- `encodingFormat`: Optional encoding format
- `dimensions`: Optional output dimensions
- `user`: Optional user identifier
- `maxConcurrentRequests`: Number of concurrent requests (default: 32)
- `batchSize`: Batch size (default: 4)
- `timeoutS`: Timeout in seconds (default: 3600)

##### `rerank(query, texts, rawScores?, returnText?, truncate?, truncationDirection?, maxConcurrentRequests?, batchSize?, timeoutS?)`

Process reranking requests.

- `query`: Query string
- `texts`: Array of texts to rerank
- `rawScores`: Return raw scores (default: false)
- `returnText`: Return text in results (default: false)
- `truncate`: Truncate texts (default: false)
- `truncationDirection`: "Left" or "Right" (default: "Right")
- `maxConcurrentRequests`: Number of concurrent requests (default: 32)
- `batchSize`: Batch size (default: 4)
- `timeoutS`: Timeout in seconds (default: 3600)

##### `classify(inputs, rawScores?, truncate?, truncationDirection?, maxConcurrentRequests?, batchSize?, timeoutS?)`

Process classification requests.

- `inputs`: Array of strings to classify
- `rawScores`: Return raw scores (default: false)
- `truncate`: Truncate texts (default: false)
- `truncationDirection`: "Left" or "Right" (default: "Right")
- `maxConcurrentRequests`: Number of concurrent requests (default: 32)
- `batchSize`: Batch size (default: 4)
- `timeoutS`: Timeout in seconds (default: 3600)

##### `batch_post(urlPath, payloads, maxConcurrentRequests?, timeoutS?)`

Send generic batch POST requests.

- `urlPath`: URL path for the requests
- `payloads`: Array of JSON payloads
- `maxConcurrentRequests`: Number of concurrent requests (default: 32)
- `timeoutS`: Timeout in seconds (default: 3600)

## Error Handling

The client can throw several types of errors:

- **Invalid Parameters**: Thrown for invalid input parameters (empty arrays, invalid timeouts, etc.)
- **Network Errors**: Connection issues, timeouts, DNS resolution failures
- **HTTP Errors**: Authentication failures (403), server errors (5xx), not found (404)
- **Serialization Errors**: Invalid JSON data

```javascript
try {
    const response = client.embed(
        ["Hello world"],
        "my_model",
        null, null, null, 32, 4, 60
    );
    console.log("Success:", response);
} catch (error) {
    console.error("Error:", error.message);
}
```

## Performance

This client is designed for high-performance scenarios and can handle thousands of concurrent requests efficiently. The Rust implementation with tokio provides excellent performance characteristics compared to pure JavaScript implementations.

## Development

### Building from Source

Requirements:
- Node.js 14+
- Rust toolchain
- napi-rs CLI

```bash
# Install dependencies
npm install

# Build the native module
npm run build

# Run tests
npm test
```

### Testing

```bash
# Run tests
npm test

# Run specific test
node test/embeddings.test.js
```

## License

MIT License

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests.
