# Integration Tests

These tests require a running test server. You can start the test server using:

```bash
cd python_bindings
python -m pytest tests/integration_fastapi_server.py::test_server -s
```

Then run the integration tests:

```bash
cargo test --package baseten-performance-reverse-proxy --test integration
```

## Test Scenarios

1. **End-to-End Embeddings Request**: Test complete flow from client to proxy to server
2. **Header Processing**: Test extraction and parsing of all custom headers
3. **Error Propagation**: Test that server errors are properly propagated
4. **Performance Metrics**: Test that timing information is preserved
5. **Concurrent Requests**: Test handling of multiple simultaneous requests
6. **Large Payloads**: Test handling of requests that exceed batch limits
