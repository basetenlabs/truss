# Baseten Performance Reverse Proxy

A high-performance performance proxy for Baseten APIs that leverages the `baseten-performance-client-core` library for optimal request processing.

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

- `Authorization: Bearer <api_key>` - API key for authentication (overrides upstream API key)
- `X-Baseten-Model: <model_name>` - Model to use for the request

### API Key Resolution

The performance proxy uses API keys in the following priority order:
1. **Request Header**: `Authorization: Bearer <api_key>` (highest priority)
2. **Upstream API Key**: `--upstream-api-key` CLI argument (fallback)

If no API key is provided in either location, the request will be rejected with `401 Unauthorized`.

### Optional Headers

- `X-Target-Host: <url>` - Override target URL for this specific request
- `X-Baseten-Request-Preferences: {...}` - JSON configuration for request processing
- `X-Baseten-Customer-Request-Id: <request_id>` - Customer request ID for tracking

## Request Processing Preferences

The `X-Baseten-Request-Preferences` header accepts JSON with the following fields:

```json
{
  "target_host": "https://api.example.com",
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

### With Target URL Override

```bash
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Baseten-Model: text-embedding-ada-002" \
  -H "X-Target-Host: https://custom.api.com" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

### With Target Host in Preferences

```bash
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Baseten-Model: text-embedding-ada-002" \
  -H "X-Baseten-Request-Preferences: {\"target_host\": \"https://api.from-preferences.com\"}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

### With Multiple Preferences Including Target Host

```bash
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Baseten-Model: text-embedding-ada-002" \
  -H "X-Baseten-Request-Preferences: {\"target_host\": \"https://api.example.com\", \"max_concurrent_requests\": 128, \"timeout_s\": 60.0}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

### Without Default Target URL

```bash
# Start proxy without default target URL
./target/release/baseten-performance-proxy --port 8080

# Each request must provide target URL
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Baseten-Model: text-embedding-ada-002" \
  -H "X-Target-Host: https://api.baseten.co" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world"],
    "encoding_format": "float"
  }'
```

## Building and Running

### From Source

```bash
# Build the performance proxy
cargo build --release --bin baseten-performance-proxy

# Run with default target URL and upstream API key
./target/release/baseten-performance-proxy \
  --target-url https://api.baseten.co \
  --upstream-api-key your-upstream-api-key

# Run with upstream API key from file (starts with /)
./target/release/baseten-performance-proxy \
  --target-url https://api.baseten.co \
  --upstream-api-key /path/to/api-key.txt

# Run without default target URL (must be provided per request)
./target/release/baseten-performance-proxy \
  --port 8080 \
  --upstream-api-key your-upstream-api-key

# Run with custom settings
./target/release/baseten-performance-proxy \
  --port 8080 \
  --target-url https://api.baseten.co \
  --upstream-api-key your-upstream-api-key \
  --max-concurrent-requests 128 \
  --batch-size 64 \
  --timeout-s 60.0
```

### API Key File Support

The `--upstream-api-key` argument supports reading API keys from files for better security:

```bash
# Create API key file
echo "your-secret-api-key" > /path/to/api-key.txt

# Use file path (starts with /)
./target/release/baseten-performance-proxy \
  --upstream-api-key /path/to/api-key.txt \
  --target-url https://api.baseten.co
```

**Security Benefits:**
- API keys not visible in process list
- Can set proper file permissions (`chmod 600 api-key.txt`)
- Supports environment variable expansion
- Works with Docker secrets and Kubernetes secrets

### With Docker

```bash
# Build the Docker image
docker build -t baseten/performance-proxy -f reverse_proxy/Dockerfile .

# Run with default settings (no default target URL)
docker run -p 8080:8080 baseten/performance-proxy

# Run with upstream API key from file
docker run -p 8080:8080 \
  -v $(pwd)/api-key.txt:/etc/baseten/api-key.txt:ro \
  baseten/performance-proxy \
  --upstream-api-key /etc/baseten/api-key.txt \
  --target-url https://api.baseten.co

# Run with environment variables
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e LOG_LEVEL=debug \
  -e HTTP_VERSION=2 \
  baseten/performance-proxy

# Run with Docker Compose
docker-compose up baseten-performance-proxy
```

### Pushing to Docker Registry

```bash
# Tag the image for the default registry location
docker tag baseten/performance-proxy baseten/performance-proxy:latest

# Push to the default registry location
docker push baseten/performance-proxy:latest

# Push with a specific version tag
docker tag baseten/performance-proxy baseten/performance-proxy:v1.0.0
docker push baseten/performance-proxy:v1.0.0
```

**Note:** The default Docker registry location for this proxy is `baseten/performance-proxy`. When pushing to a registry, always use this naming convention to ensure consistency.

### Docker Compose Examples

```yaml
# Basic usage (API key must be provided per request)
version: '3.8'
services:
  proxy:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - LOG_LEVEL=info

# With upstream API key
version: '3.8'
services:
  proxy:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./api-key.txt:/etc/baseten/api-key.txt:ro
    command:
      - "--upstream-api-key"
      - "/etc/baseten/api-key.txt"
      - "--target-url"
      - "https://api.baseten.co"
```

## CLI Options

```
baseten-performance-proxy [OPTIONS]

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
  "service": "baseten-performance-proxy",
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
  performance-proxy:
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
  name: baseten-performance-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: baseten-performance-proxy
  template:
    metadata:
      labels:
        app: baseten-performance-proxy
    spec:
      containers:
      - name: performance-proxy
        image: baseten/performance-proxy:latest
        ports:
        - containerPort: 8080
        command:
        - baseten-performance-proxy
        - --target-url=https://api.baseten.co
        - --port=8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Truss Configuration

For deploying with Truss, use the following configuration in your `config.yaml`:

```yaml
base_image:
  image: baseten/performance-proxy:0.0.1
docker_server:
  start_command: sh -c "baseten-performance-proxy --port 8081 --upstream-api-key $/secrets/upstream_api_key --target-url ${UPSTREAM_URL} --tokenizer BAAI/bge-small-en-v1.5 /app/tokenizers/bge-small-en-v1.5/tokenizer.json --tokenizer voyageai/voyage-4-nano /app/tokenizers/voyage-4-nano/tokenizer.json --http-version 2 --max-chars-per-request 10000 --timeout-s 300 --batch-size 16 --max-concurrent-requests 64"
  readiness_endpoint: /health_internal
  liveness_endpoint: /health_internal
  predict_endpoint: /v1/embeddings
  server_port: 8081
build_commands:
# Download tokenizers for multiple models
- sh -c "mkdir -p /app/tokenizers/bge-small-en-v1.5 && cd /app/tokenizers/bge-small-en-v1.5 && wget https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json -O tokenizer.json"
- sh -c "mkdir -p /app/tokenizers/voyage-4-nano && cd /app/tokenizers/voyage-4-nano && wget https://huggingface.co/voyageai/voyage-4-nano/resolve/main/tokenizer.json -O tokenizer.json"
resources:
  use_gpu: false
model_name: baseten-performance-proxy
environment_variables:
  UPSTREAM_URL: https://model-abcdefg.api.baseten.co/environments/production/sync
  PERFORMANCE_CLIENT_LOG_LEVEL: info
secrets:
  upstream_api_key: null # name saved in baseten-ui
```

## License

MIT License - see LICENSE file for details.
