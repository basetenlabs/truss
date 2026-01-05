#!/usr/bin/env python3
"""
Concise parameterized test for HTTP method functionality
"""

import pytest
from baseten_performance_client import PerformanceClient


@pytest.mark.parametrize(
    "method,expected_path",
    [
        ("GET", "/get"),
        ("POST", "/post"),
        ("PUT", "/put"),
        ("PATCH", "/patch"),
        ("DELETE", "/delete"),
        ("HEAD", "/get"),  # HEAD to /get returns headers only
        ("OPTIONS", "/"),  # OPTIONS to root returns CORS headers
        (None, "/post"),  # Default to POST
    ],
)
def test_http_methods(method, expected_path):
    """Test all HTTP methods including default behavior"""
    client = PerformanceClient(base_url="https://httpbin.org", api_key="test-key")
    payloads = [{"test": "data"}]

    kwargs = {"method": method} if method else {}
    response = client.batch_post(expected_path, payloads, **kwargs)

    assert len(response.data) == 1
    assert response.total_time > 0


def test_invalid_method():
    """Test that invalid methods are rejected"""
    client = PerformanceClient(base_url="https://httpbin.org", api_key="test-key")

    with pytest.raises(ValueError, match="Invalid HTTP method"):
        client.batch_post("/post", [{"test": "data"}], method="INVALID")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
