#!/bin/bash

# Integration test runner for the reverse proxy
# This script runs comprehensive integration tests to ensure the proxy works correctly

set -e

echo "🧪 Starting Reverse Proxy Integration Tests"
echo "=========================================="

# Build the reverse proxy
echo "📦 Building reverse proxy..."
cargo build --release --package baseten-performance-reverse-proxy

# Run unit tests
echo "🔬 Running unit tests..."
cargo test --package baseten-performance-reverse-proxy

# Run integration tests
echo "🔗 Running integration tests..."
cargo test --package baseten-performance-reverse-proxy --test integration_test -- --nocapture

echo ""
echo "✅ All tests completed successfully!"
echo "=========================================="

# Test scenarios that will be run:
echo ""
echo "📋 Test Scenarios:"
echo "  1. Basic embeddings request through proxy"
echo "  2. Large batch request (100 items)"
echo "  3. Request with custom preferences"
echo "  4. Error handling through proxy"
echo "  5. Concurrent requests (10 parallel)"
echo "  6. Rerank endpoint through proxy"
echo "  7. Classify endpoint through proxy"
echo "  8. Generic batch endpoint through proxy"
echo ""
echo "🎯 Each scenario tests:"
echo "  - Request forwarding through proxy"
echo "  - Header handling (API key, model, customer ID)"
echo "  - Response metadata and timing"
echo "  - Error propagation"
echo "  - Performance characteristics"
echo ""
echo "🚀 To run tests manually:"
echo "  cargo test --package baseten-performance-reverse-proxy --test integration_test"
echo ""
echo "📊 Test Architecture:"
echo "  - Mock server on port 8082"
echo "  - Reverse proxy on port 8081"
echo "  - Core client connecting through proxy"
echo "  - Comprehensive scenario testing"
