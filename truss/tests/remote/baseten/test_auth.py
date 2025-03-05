import pytest

from truss.remote.baseten.auth import ApiKey, AuthService
from truss.remote.baseten.error import AuthorizationError


def test_api_key():
    key = ApiKey("test_key")
    assert key.value == "test_key"
    assert key.header() == {"Authorization": "Api-Key test_key"}


def test_auth_service_no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    auth_service = AuthService()
    with pytest.raises(AuthorizationError):
        auth_service.authenticate()


def test_auth_service_with_key():
    auth_service = AuthService("test_key")
    key = auth_service.authenticate()
    assert key.value == "test_key"
