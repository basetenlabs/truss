import time

import pydantic
import pytest

from truss.remote.baseten import oauth as oauth_mod
from truss.remote.baseten.auth import (
    ApiKeyCredential,
    AuthService,
    OAuthSession,
    WorkloadIdentityCredential,
    workload_identity_from_env,
)


def test_api_key_auth_header():
    auth_service = AuthService(ApiKeyCredential(api_key="test_key"))
    assert auth_service.fetch_auth_header() == {"Authorization": "Api-Key test_key"}


def test_api_key_auth_header_use_bearer():
    auth_service = AuthService(ApiKeyCredential(api_key="test_key", use_bearer=True))
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer test_key"}


def test_api_key_credential_rejects_empty():
    with pytest.raises(pydantic.ValidationError):
        ApiKeyCredential(api_key="")


def test_oauth_auth_header_is_bearer():
    cred = oauth_mod.OAuthCredential(
        access_token="at", refresh_token="rt", expires_at=int(time.time()) + 3600
    )
    auth_service = AuthService(
        OAuthSession(api_url="https://api.baseten.co", credential=cred)
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at"}


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
        OAuthSession(
            api_url="https://api.baseten.co",
            credential=cred,
            on_token_refresh=persisted.append,
        )
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer new"}
    assert len(persisted) == 1
    assert persisted[0].access_token == "new"


def _workload_identity_service(tmp_path, monkeypatch, expires_in_seconds=3600):
    token_file = tmp_path / "token"
    token_file.write_text("jwt-1")
    exchanges = []

    def fake_exchange(api_url, identity_token_file, trust_rule_id):
        exchanges.append((api_url, identity_token_file.read_text(), trust_rule_id))
        return oauth_mod.AccessToken(
            access_token=f"at-{len(exchanges)}",
            expires_at=int(time.time()) + expires_in_seconds,
        )

    monkeypatch.setattr(
        "truss.remote.baseten.auth.exchange_identity_token", fake_exchange
    )
    auth_service = AuthService(
        WorkloadIdentityCredential(
            api_url="https://api.baseten.co",
            identity_token_file=token_file,
            trust_rule_id="rule-1",
        )
    )
    return auth_service, token_file, exchanges


def test_workload_identity_exchanges_once_and_caches(tmp_path, monkeypatch):
    auth_service, _, exchanges = _workload_identity_service(tmp_path, monkeypatch)
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    assert exchanges == [("https://api.baseten.co", "jwt-1", "rule-1")]


def test_workload_identity_re_exchanges_near_expiry_with_rotated_token(
    tmp_path, monkeypatch
):
    # Inside the refresh leeway, so every call exchanges again.
    auth_service, token_file, exchanges = _workload_identity_service(
        tmp_path, monkeypatch, expires_in_seconds=oauth_mod.EXPIRY_LEEWAY_SECONDS - 1
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    token_file.write_text("jwt-2")
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-2"}
    assert [assertion for _, assertion, _ in exchanges] == ["jwt-1", "jwt-2"]


@pytest.mark.parametrize(
    "env",
    [{"BASETEN_IDENTITY_TOKEN_FILE": "/token"}, {"BASETEN_TRUST_RULE_ID": "rule-1"}],
)
def test_workload_identity_from_env_half_configured_raises(monkeypatch, env):
    monkeypatch.delenv("BASETEN_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("BASETEN_TRUST_RULE_ID", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="needs both"):
        workload_identity_from_env()
