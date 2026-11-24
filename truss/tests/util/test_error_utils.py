import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from truss.util.error_utils import handle_client_error


def test_no_error_passes_through():
    with handle_client_error("uploading"):
        result = 1 + 1
    assert result == 2


def test_no_credentials_error_is_wrapped():
    original = NoCredentialsError()
    with pytest.raises(
        RuntimeError, match="No AWS credentials found for uploading"
    ) as exc_info:
        with handle_client_error("uploading"):
            raise original
    assert exc_info.value.__cause__ is original


def test_client_error_is_wrapped():
    original = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "PutObject"
    )
    with pytest.raises(
        RuntimeError, match="AWS client error when uploading"
    ) as exc_info:
        with handle_client_error("uploading"):
            raise original
    assert "AccessDenied" in str(exc_info.value)
    assert exc_info.value.__cause__ is original


def test_unexpected_error_is_wrapped_without_repeating_message():
    original = ValueError("disk full")
    with pytest.raises(RuntimeError) as exc_info:
        with handle_client_error("uploading"):
            raise original
    message = str(exc_info.value)
    assert message == "Unexpected error during `uploading`: disk full"
    assert message.count("disk full") == 1
    assert exc_info.value.__cause__ is original
