def test_baseten_performance_client_bindings_basic_test():
    from baseten_performance_client import PerformanceClient

    PerformanceClient.embed
    PerformanceClient.async_embed
    PerformanceClient.rerank
    PerformanceClient.async_rerank
    PerformanceClient.classify
    PerformanceClient.async_classify
    PerformanceClient.batch_post
    PerformanceClient.async_batch_post


def test_http_client_wrapper_initialization():
    """Test that HttpClientWrapper can be initialized with different http_version values."""
    from baseten_performance_client import HttpClientWrapper

    # Test default initialization
    wrapper = HttpClientWrapper()
    assert wrapper is not None

    # Test with http_version=1 (HTTP/1.1)
    wrapper1 = HttpClientWrapper(http_version=1)
    assert wrapper1 is not None

    # Test with http_version=2 (HTTP/2)
    wrapper2 = HttpClientWrapper(http_version=2)
    assert wrapper2 is not None


def test_performance_client_with_http_client_wrapper():
    """Test that PerformanceClient can be initialized with an HttpClientWrapper."""
    from baseten_performance_client import HttpClientWrapper, PerformanceClient

    wrapper = HttpClientWrapper(http_version=1)
    client = PerformanceClient(
        base_url="https://api.example.com",
        api_key="test-api-key",
        http_version=1,
        client_wrapper=wrapper,
    )
    assert client is not None


def test_get_client_wrapper():
    """Test that get_client_wrapper returns an HttpClientWrapper."""
    from baseten_performance_client import PerformanceClient

    client = PerformanceClient(
        base_url="https://api.example.com", api_key="test-api-key"
    )
    wrapper = client.get_client_wrapper()
    assert wrapper is not None


def test_sharing_http_client_wrapper_between_clients():
    """Test that HttpClientWrapper can be shared between multiple clients."""
    from baseten_performance_client import HttpClientWrapper, PerformanceClient

    wrapper = HttpClientWrapper(http_version=1)

    client1 = PerformanceClient(
        base_url="https://api1.example.com",
        api_key="test-api-key-1",
        client_wrapper=wrapper,
    )
    client2 = PerformanceClient(
        base_url="https://api2.example.com",
        api_key="test-api-key-2",
        client_wrapper=wrapper,
    )

    assert client1 is not None
    assert client2 is not None


def test_get_wrapper_from_client_and_reuse():
    """Test getting wrapper from one client and using it in another."""
    from baseten_performance_client import PerformanceClient

    client1 = PerformanceClient(
        base_url="https://api1.example.com", api_key="test-api-key-1"
    )
    wrapper = client1.get_client_wrapper()

    client2 = PerformanceClient(
        base_url="https://api2.example.com",
        api_key="test-api-key-2",
        client_wrapper=wrapper,
    )
    assert client2 is not None
