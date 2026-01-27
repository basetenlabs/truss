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


def test_request_processing_preference_basic():
    """Test RequestProcessingPreference basic functionality."""
    from baseten_performance_client import RequestProcessingPreference

    # Test default creation
    preference = RequestProcessingPreference()
    assert preference is not None
    assert preference.max_concurrent_requests > 0
    assert preference.batch_size > 0
    assert preference.timeout_s > 0

    # Test class method default
    default_preference = RequestProcessingPreference.default()
    assert default_preference is not None
    assert default_preference.max_concurrent_requests > 0

    # Test custom parameters
    custom_preference = RequestProcessingPreference(
        max_concurrent_requests=64,
        batch_size=32,
        timeout_s=30.0,
        hedge_delay=0.5,
        total_timeout_s=60.0,
    )
    assert custom_preference.max_concurrent_requests == 64
    assert custom_preference.batch_size == 32
    assert custom_preference.timeout_s == 30.0
    assert custom_preference.hedge_delay == 0.5
    assert custom_preference.total_timeout_s == 60.0


def test_request_processing_preference_property_setters():
    """Test RequestProcessingPreference property setters."""
    from baseten_performance_client import RequestProcessingPreference

    preference = RequestProcessingPreference()

    # Test property setters
    preference.max_concurrent_requests = 128
    preference.batch_size = 64
    preference.timeout_s = 45.0
    preference.hedge_delay = 1.0
    preference.total_timeout_s = 90.0
    preference.hedge_budget_pct = 0.15
    preference.retry_budget_pct = 0.10
    preference.max_retries = 5
    preference.initial_backoff_ms = 250

    assert preference.max_concurrent_requests == 128
    assert preference.batch_size == 64
    assert preference.timeout_s == 45.0
    assert preference.hedge_delay == 1.0
    assert preference.total_timeout_s == 90.0
    assert preference.hedge_budget_pct == 0.15
    assert preference.retry_budget_pct == 0.10
    assert preference.max_retries == 5
    assert preference.initial_backoff_ms == 250


def test_cancellation_token_basic():
    """Test CancellationToken basic functionality."""
    from baseten_performance_client import CancellationToken

    # Test creation
    token = CancellationToken()
    assert token is not None
    assert not token.is_cancelled()

    # Test cancellation
    token.cancel()
    assert token.is_cancelled()


def test_performance_client_with_preference():
    """Test PerformanceClient methods with RequestProcessingPreference."""
    from baseten_performance_client import (
        PerformanceClient,
        RequestProcessingPreference,
    )

    client = PerformanceClient(
        base_url="https://api.example.com", api_key="test-api-key"
    )

    _preference = RequestProcessingPreference(
        max_concurrent_requests=32, batch_size=16, timeout_s=60.0
    )

    # Test that methods accept preference parameter (without making actual requests)
    assert hasattr(client, "embed")
    assert hasattr(client, "async_embed")
    assert hasattr(client, "rerank")
    assert hasattr(client, "async_rerank")
    assert hasattr(client, "classify")
    assert hasattr(client, "async_classify")
    assert hasattr(client, "batch_post")
    assert hasattr(client, "async_batch_post")


def test_performance_client_with_cancellation_token():
    """Test PerformanceClient with CancellationToken in preference."""
    from baseten_performance_client import (
        CancellationToken,
        PerformanceClient,
        RequestProcessingPreference,
    )

    _client = PerformanceClient(
        base_url="https://api.example.com", api_key="test-api-key"
    )

    token = CancellationToken()
    assert token.is_cancelled() is False
    preference = RequestProcessingPreference(
        max_concurrent_requests=32, cancel_token=token
    )
    assert token.is_cancelled() is False
    assert preference.cancel_token is not None
    assert not preference.cancel_token.is_cancelled()

    # Test cancellation
    preference.cancel_token.cancel()
    assert preference.cancel_token.is_cancelled()
    assert token.is_cancelled()


def test_request_processing_preference_api_key_override():
    """Test RequestProcessingPreference with API key override."""
    from baseten_performance_client import RequestProcessingPreference

    # Test default creation (no API key override)
    preference = RequestProcessingPreference()
    assert preference.primary_api_key_override is None

    # Test creation with API key override
    override_key = "override-api-key-12345"
    preference_with_override = RequestProcessingPreference(
        primary_api_key_override=override_key
    )
    assert preference_with_override.primary_api_key_override == override_key

    # Test setting API key override after creation
    preference.primary_api_key_override = "new-override-key"
    assert preference.primary_api_key_override == "new-override-key"

    # Test with other parameters and API key override
    complex_preference = RequestProcessingPreference(
        max_concurrent_requests=64,
        batch_size=32,
        timeout_s=30.0,
        primary_api_key_override="complex-override-key",
    )
    assert complex_preference.max_concurrent_requests == 64
    assert complex_preference.batch_size == 32
    assert complex_preference.timeout_s == 30.0
    assert complex_preference.primary_api_key_override == "complex-override-key"


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
