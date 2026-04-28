import time

import pytest

from truss.remote.baseten import oauth as oauth_mod
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.error import AuthorizationError


def test_api_key_auth_header():
    auth_service = AuthService("test_key")
    assert auth_service.auth_header() == {"Authorization": "Api-Key test_key"}


def test_auth_service_no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    auth_service = AuthService()
    with pytest.raises(AuthorizationError):
        auth_service.auth_header()


def test_oauth_auth_header_is_bearer():
    cred = oauth_mod.OAuthCredential(
        access_token="at", refresh_token="rt", expires_at=int(time.time()) + 3600
    )
    auth_service = AuthService(api_url="https://api.baseten.co", oauth_credential=cred)
    assert auth_service.auth_header() == {"Authorization": "Bearer at"}


def test_oauth_refresh_when_expired(monkeypatch):
    cred = oauth_mod.OAuthCredential(
        access_token="old", refresh_token="rt-old", expires_at=int(time.time()) - 1
    )
    new_cred = oauth_mod.OAuthCredential(
        access_token="new", refresh_token="rt-new", expires_at=int(time.time()) + 3600
    )
    persisted = []

    def fake_refresh(api_url, credential):
        assert credential.refresh_token == "rt-old"
        return new_cred

    monkeypatch.setattr("truss.remote.baseten.auth.refresh", fake_refresh)
    auth_service = AuthService(
        api_url="https://api.baseten.co",
        oauth_credential=cred,
        on_token_refresh=persisted.append,
    )
    assert auth_service.auth_header() == {"Authorization": "Bearer new"}
    assert len(persisted) == 1
    assert persisted[0].access_token == "new"
