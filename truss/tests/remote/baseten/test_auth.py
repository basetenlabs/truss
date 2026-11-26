import re
import time

import pydantic
import pytest
import requests

from truss.remote.baseten import oauth as oauth_mod
from truss.remote.baseten.auth import (
    ApiKeyCredential,
    AuthService,
    OAuthSession,
    WorkloadIdentityConfig,
    WorkloadIdentityCredential,
    refused_by_baseten,
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


def _workload_identity_service(
    tmp_path, monkeypatch, expires_in_seconds=3600, lifetime_seconds=3600
):
    token_file = tmp_path / "token"
    token_file.write_text("jwt-1")
    exchanges = []

    def fake_exchange(api_url, assertion, trust_rule_id, assertion_source):
        exchanges.append((api_url, assertion, trust_rule_id))
        return oauth_mod.AccessToken(
            access_token=f"at-{len(exchanges)}",
            expires_at=int(time.time()) + expires_in_seconds,
            lifetime_seconds=lifetime_seconds,
        )

    monkeypatch.setattr(
        "truss.remote.baseten.auth.exchange_identity_token", fake_exchange
    )
    auth_service = AuthService(
        WorkloadIdentityCredential(
            api_url="https://api.baseten.co",
            config=WorkloadIdentityConfig(
                trust_rule_id="rule-1", identity_token_file=token_file
            ),
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
        tmp_path, monkeypatch, expires_in_seconds=100
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    token_file.write_text("jwt-2")
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-2"}
    assert [assertion for _, assertion, _ in exchanges] == ["jwt-1", "jwt-2"]


def test_workload_identity_short_lived_token_is_reused(tmp_path, monkeypatch):
    # The five-minute leeway is capped at half the lifetime, so a two-minute
    # token is reused for its first minute.
    auth_service, _, exchanges = _workload_identity_service(
        tmp_path, monkeypatch, expires_in_seconds=120, lifetime_seconds=120
    )
    auth_service.fetch_auth_header()
    auth_service.fetch_auth_header()
    assert len(exchanges) == 1


def test_workload_identity_inline_token_is_trimmed_and_must_not_be_empty(monkeypatch):
    assertions = []

    def fake_exchange(api_url, assertion, trust_rule_id, assertion_source):
        assertions.append(assertion)
        return oauth_mod.AccessToken(
            access_token="at-1", expires_at=int(time.time()) + 3600
        )

    monkeypatch.setattr(
        "truss.remote.baseten.auth.exchange_identity_token", fake_exchange
    )
    for token, expected in (("  jwt \n", ["jwt"]), (" \n", [])):
        assertions.clear()
        auth_service = AuthService(
            WorkloadIdentityCredential(
                api_url="https://api.baseten.co",
                config=WorkloadIdentityConfig(
                    trust_rule_id="rule-1", identity_token=token
                ),
            )
        )
        if expected:
            auth_service.fetch_auth_header()
        else:
            with pytest.raises(oauth_mod.OAuthError, match="is empty"):
                auth_service.fetch_auth_header()
        assert assertions == expected


def test_workload_identity_failed_refresh_keeps_unexpired_token(tmp_path, monkeypatch):
    auth_service, _, _ = _workload_identity_service(
        tmp_path, monkeypatch, expires_in_seconds=100
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}

    def failing_exchange(*args):
        raise oauth_mod.OAuthError("exchange failed")

    monkeypatch.setattr(
        "truss.remote.baseten.auth.exchange_identity_token", failing_exchange
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}


def test_workload_identity_failed_refresh_raises_once_expired(tmp_path, monkeypatch):
    auth_service, _, _ = _workload_identity_service(
        tmp_path, monkeypatch, expires_in_seconds=0
    )
    auth_service.fetch_auth_header()

    def failing_exchange(*args):
        raise oauth_mod.OAuthError("exchange failed")

    monkeypatch.setattr(
        "truss.remote.baseten.auth.exchange_identity_token", failing_exchange
    )
    with pytest.raises(oauth_mod.OAuthError, match="exchange failed"):
        auth_service.fetch_auth_header()


def test_workload_identity_unreadable_file_raises_before_exchange(
    tmp_path, monkeypatch
):
    auth_service, token_file, exchanges = _workload_identity_service(
        tmp_path, monkeypatch
    )
    token_file.write_text("  \n")
    with pytest.raises(oauth_mod.OAuthError, match="is empty"):
        auth_service.fetch_auth_header()
    token_file.unlink()
    with pytest.raises(oauth_mod.OAuthError, match="Cannot read"):
        auth_service.fetch_auth_header()
    assert exchanges == []


def test_workload_identity_inline_token(monkeypatch):
    exchanges = []

    def fake_exchange(api_url, assertion, trust_rule_id, assertion_source):
        exchanges.append((assertion, trust_rule_id, assertion_source))
        return oauth_mod.AccessToken(
            access_token="at-1", expires_at=int(time.time()) + 3600
        )

    monkeypatch.setattr(
        "truss.remote.baseten.auth.exchange_identity_token", fake_exchange
    )
    _clear_workload_identity_env(monkeypatch)
    monkeypatch.setenv("BASETEN_WORKLOAD_IDENTITY_TOKEN", "jwt-inline")
    monkeypatch.setenv("BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID", "rule-1")
    config = workload_identity_from_env()
    assert "jwt-inline" not in repr(config)
    auth_service = AuthService(
        WorkloadIdentityCredential(api_url="https://api.baseten.co", config=config)
    )
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    assert exchanges == [("jwt-inline", "rule-1", "BASETEN_WORKLOAD_IDENTITY_TOKEN")]


def _clear_workload_identity_env(monkeypatch):
    for name in (
        "BASETEN_API_KEY",
        "BASETEN_TRUSS_AUTH_API_KEY",
        "BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE",
        "BASETEN_WORKLOAD_IDENTITY_TOKEN",
        "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "env, message",
    [
        (
            {"BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE": "/token"},
            "BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE is set but "
            "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID is not",
        ),
        (
            {"BASETEN_WORKLOAD_IDENTITY_TOKEN": "jwt"},
            "BASETEN_WORKLOAD_IDENTITY_TOKEN is set but "
            "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID is not",
        ),
        (
            {"BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID": "rule-1"},
            "needs an identity token",
        ),
        (
            {
                "BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE": "/token",
                "BASETEN_WORKLOAD_IDENTITY_TOKEN": "jwt",
                "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID": "rule-1",
            },
            "Both BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE and "
            "BASETEN_WORKLOAD_IDENTITY_TOKEN are set",
        ),
        (
            {
                "BASETEN_API_KEY": "key",
                "BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE": "/token",
                "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID": "rule-1",
            },
            "Both BASETEN_API_KEY and workload identity",
        ),
        (
            {
                "BASETEN_TRUSS_AUTH_API_KEY": "key",
                "BASETEN_WORKLOAD_IDENTITY_TOKEN": "jwt",
                "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID": "rule-1",
            },
            "Both BASETEN_TRUSS_AUTH_API_KEY and workload identity",
        ),
    ],
)
def test_workload_identity_from_env_ambiguous_or_incomplete_raises(
    monkeypatch, env, message
):
    _clear_workload_identity_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=re.escape(message)):
        workload_identity_from_env()


def _response(status_code, challenge, authorization):
    response = requests.Response()
    response.status_code = status_code
    if challenge is not None:
        response.headers["WWW-Authenticate"] = challenge
    response.request = requests.Request(
        "GET",
        "https://api.baseten.co/v1/models",
        headers={"Authorization": authorization},
    ).prepare()
    return response


REFUSED_CHALLENGE = 'Bearer realm="baseten", error="invalid_token"'


def test_workload_identity_refused_token_is_replaced_on_next_request(
    tmp_path, monkeypatch
):
    auth_service, _, exchanges = _workload_identity_service(tmp_path, monkeypatch)
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    auth_service.note_response(_response(401, REFUSED_CHALLENGE, "Bearer at-1"))
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-2"}
    assert len(exchanges) == 2


@pytest.mark.parametrize(
    "status_code, challenge, authorization",
    [
        # A model's own 401 carries no Baseten challenge.
        (401, None, "Bearer at-1"),
        (401, 'Bearer realm="model", error="invalid_token"', "Bearer at-1"),
        (403, REFUSED_CHALLENGE, "Bearer at-1"),
        # A refusal of an older token never discards the current one.
        (401, REFUSED_CHALLENGE, "Bearer at-0"),
    ],
)
def test_workload_identity_token_kept_unless_baseten_refused_it(
    tmp_path, monkeypatch, status_code, challenge, authorization
):
    auth_service, _, exchanges = _workload_identity_service(tmp_path, monkeypatch)
    auth_service.fetch_auth_header()
    auth_service.note_response(_response(status_code, challenge, authorization))
    assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-1"}
    assert len(exchanges) == 1


def test_refused_by_baseten_finds_challenge_among_several():
    challenge = 'Bearer realm="model", Bearer realm="baseten", error="invalid_token"'
    assert refused_by_baseten(_response(401, challenge, "Bearer x"))


def test_rest_and_graphql_report_refusals(tmp_path, monkeypatch):
    import requests_mock

    from truss.remote.baseten.api import BasetenApi

    auth_service, _, exchanges = _workload_identity_service(tmp_path, monkeypatch)
    api = BasetenApi("https://app.baseten.co", auth_service)
    refused = {"status_code": 401, "headers": {"WWW-Authenticate": REFUSED_CHALLENGE}}
    with requests_mock.Mocker() as m:
        m.get("https://api.baseten.co/v1/models", **refused)
        m.post("https://app.baseten.co/graphql/", **refused)
        with pytest.raises(requests.HTTPError):
            api._rest_api_client.get("v1/models")
        assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-2"}
        with pytest.raises(requests.HTTPError):
            api._post_graphql_query("{ me { id } }")
        assert auth_service.fetch_auth_header() == {"Authorization": "Bearer at-3"}
    assert len(exchanges) == 3
