from unittest.mock import MagicMock

from truss.remote.baseten import service
from truss.remote.baseten.core import ModelVersionHandle


def test_model_invoke_url_prod():
    url = service.URLConfig.invoke_url(
        "https://model-123.api.baseten.co",
        service.URLConfig.MODEL,
        "789",
        is_draft=False,
    )
    assert url == "https://model-123.api.baseten.co/deployment/789/predict"


def test_model_invoke_url_draft():
    url = service.URLConfig.invoke_url(
        "https://model-123.api.baseten.co",
        service.URLConfig.MODEL,
        "789",
        is_draft=True,
    )
    assert url == "https://model-123.api.baseten.co/development/predict"


def test_chain_invoke_url_prod():
    url = service.URLConfig.invoke_url(
        "https://chain-abc.api.baseten.co",
        service.URLConfig.CHAIN,
        "666",
        is_draft=False,
    )
    assert url == "https://chain-abc.api.baseten.co/deployment/666/run_remote"


def test_chain_invoke_url_draft():
    url = service.URLConfig.invoke_url(
        "https://chain-abc.api.baseten.co",
        service.URLConfig.CHAIN,
        "666",
        is_draft=True,
    )
    assert url == "https://chain-abc.api.baseten.co/development/run_remote"


def test_model_status_page_url():
    url = service.URLConfig.status_page_url(
        "https://app.baseten.co", service.URLConfig.MODEL, "123"
    )
    assert url == "https://app.baseten.co/models/123/overview"


def test_chain_status_page_url():
    url = service.URLConfig.status_page_url(
        "https://app.baseten.co", service.URLConfig.CHAIN, "abc"
    )
    assert url == "https://app.baseten.co/chains/abc/overview"


def test_model_logs_url():
    url = service.URLConfig.model_logs_url("https://app.baseten.co", "123", "789")
    assert url == "https://app.baseten.co/models/123/logs/789"


def test_chain_logs_url():
    url = service.URLConfig.chainlet_logs_url(
        "https://app.baseten.co", "abc", "666", "543"
    )
    assert url == "https://app.baseten.co/chains/abc/logs/666/543"


def test_predict_response_to_json():
    """Test that predict method returns JSON response for normal dict result."""
    # Create a mock BasetenService
    mock_handle = MagicMock(spec=ModelVersionHandle)
    mock_handle.model_id = "test-model"
    mock_handle.version_id = "test-version"
    mock_handle.hostname = "https://model-test.api.baseten.co"

    mock_api = MagicMock()
    mock_api.app_url = "https://app.baseten.co"

    service_instance = service.BasetenService(
        model_version_handle=mock_handle,
        is_draft=False,
        api_key="test-key",
        service_url="https://test.com",
        api=mock_api,
    )

    # Mock the _send_request method to return a successful JSON response
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": "success"}
    service_instance._send_request = MagicMock(return_value=mock_response)

    # Test predict method
    result = service_instance.predict({"input": "test"})

    # Verify that the JSON response is returned directly
    assert result == {"result": "success"}

    # Test non-dict response types below

    # With integer response
    mock_response.json.return_value = 42
    result = service_instance.predict({"input": "test"})
    assert result == 42

    # With string response
    mock_response.json.return_value = "success"
    result = service_instance.predict({"input": "test"})
    assert result == "success"

    # With list response
    mock_response.json.return_value = [1, 2, 3, 4]
    result = service_instance.predict({"input": "test"})
    assert result == [1, 2, 3, 4]

    # With boolean response
    mock_response.json.return_value = True
    result = service_instance.predict({"input": "test"})
    assert result is True
