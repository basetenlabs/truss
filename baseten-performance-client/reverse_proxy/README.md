# Baseten Performance Reverse Proxy

A high-performance reverse proxy for Baseten APIs that leverages the `baseten-performance-client-core` library for optimal request processing.

## Features

- **Unified Handler**: Economically handles all API endpoints with a single handler
- **HTTP/1.1 & HTTP/2 Support**: Full support for both HTTP protocols
- **Header-Based Configuration**: Uses `X-Baseten-*` headers for request configuration
- **Performance Optimization**: Leverages batching, concurrency, and retry logic from the core client
- **Docker Support**: Ready-to-use Docker image for deployment

## Supported Endpoints

- `/v1/embeddings` - OpenAI-compatible embeddings API
- `/rerank` - Reranking service
- `/predict` - Classification service
- `/classify` - Alternative classification endpoint
- `/health` - Health check endpoint
- `/*path` - Generic batch requests

## Request Headers

### Required Headers

- `Authorization: Bearer <api_key>` - API key for authentication
- `X-Baseten-Model: <model_name>` - Model to use for the request

### Optional Headers

- `X-Baseten-Request-Preferences: {...}` - JSON configuration for request processing
- `X-Baseten-Customer-Request-Id: <request_id>` - Customer request ID for tracking

## Request Processing Preferences

The `X-Baseten-Request-Preferences` header accepts JSON with the following fields:

```json
{
  "max_concurrent_requests": 64,
  "batch_size": 32,
  "timeout_s": 30.0,
  "hedge_delay": 0.5,
  "hedge_budget_pct": 0.15,
  "retry_budget_pct": 0.08,
  "max_retries": 3,
  "initial_backoff_ms": 125
}
```

## Usage Examples

### Embeddings Request

```bash
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Baseten-Model: text-embedding-ada-002" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

### With Custom Preferences

```bash
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Baseten-Model: text-embedding-ada-002" \
  -H "X-Baseten-Request-Preferences: {\"max_concurrent_requests\":128,\"timeout_s\":60.0}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

### Rerank Request

```bash
curl -X POST "http://localhost:8080/rerank" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "texts": ["Paris is the capital of France", "London is the capital of England", "Berlin is the capital of Germany"]
  }'
```

## Building and Running

### From Source

```bash
# Build the reverse proxy
cargo build --release --bin baseten-reverse-proxy

# Run with default settings
./target/release/baseten-reverse-proxy --target-url https://api.baseten.co

# Run with custom settings
./target/release/baseten-reverse-proxy \
  --port 8080 \
  --target-url https://api.baseten.co \
  --max-concurrent-requests 128 \
  --batch-size 64 \
  --timeout-s 60.0
```

### With Docker

```bash
# Build the Docker image
docker build -t baseten-reverse-proxy ./reverse_proxy

# Run with default settings
docker run -p 8080:8080 baseten-reverse-proxy

# Run with custom settings
docker run -p 8080:8080 \
  baseten-reverse-proxy \
  --target-url https://api.baseten.co \
  --max-concurrent-requests 128
```

## CLI Options

```
baseten-reverse-proxy [OPTIONS]

Options:
  -p, --port <PORT>                 Port to listen on [default: 8080]
  -u, --target-url <TARGET_URL>     Target API URL
  --http-version <HTTP_VERSION>     HTTP version to use [default: 2]
  --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
                                    Maximum concurrent requests [default: 64]
  --batch-size <BATCH_SIZE>         Batch size for requests [default: 32]
  --timeout-s <TIMEOUT_S>           Request timeout in seconds [default: 30.0]
  --log-level <LOG_LEVEL>           Log level [default: info]
  -h, --help                        Print help
  -V, --version                     Print version
```

## Response Format

All responses include proxy metadata:

```json
{
  "data": [...],
  "model": "text-embedding-ada-002",
  "usage": {...},
  "proxy_metadata": {
    "total_time": 1.234,
    "batch_count": 2,
    "customer_request_id": "req_123456789",
    "response_headers": [...]
  }
}
```

## Error Handling

The proxy returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid parameters or headers
- `401 Unauthorized` - Missing or invalid API key
- `502 Bad Gateway` - Upstream service error
- `500 Internal Server Error` - Proxy internal error

Error responses include details:

```json
{
  "error": "Bad request - invalid parameters or headers",
  "path": "/v1/embeddings",
  "method": "Post"
}
```

## Performance Considerations

- The proxy leverages the same batching and concurrency logic as the core client
- HTTP/2 is enabled by default for multiplexing
- Connection pooling is handled automatically
- Request hedging and retry logic are preserved

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "service": "baseten-reverse-proxy",
  "timestamp": "2024-01-25T12:00:00Z"
}
```

### Metrics

The proxy includes timing information in all responses:
- `total_time`: Total processing time
- `batch_count`: Number of batches processed
- `individual_request_times`: Timing for each batch

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  reverse-proxy:
    build: ./reverse_proxy
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
    command:
      - --target-url=https://api.baseten.co
      - --port=8080
      - --max-concurrent-requests=128
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: baseten-reverse-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: baseten-reverse-proxy
  template:
    metadata:
      labels:
        app: baseten-reverse-proxy
    spec:
      containers:
      - name: reverse-proxy
        image: baseten-reverse-proxy:latest
        ports:
        - containerPort: 8080
        command:
        - baseten-reverse-proxy
        - --target-url=https://api.baseten.co
        - --port=8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

## License

MIT License - see LICENSE file for details.
