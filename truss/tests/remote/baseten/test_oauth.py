import time

import pytest
import requests_mock

from truss.remote.baseten import oauth

API_URL = "https://api.baseten.co"
DEVICE_AUTHORIZE_URL = API_URL + oauth.DEVICE_AUTHORIZE_PATH
DEVICE_TOKEN_URL = API_URL + oauth.DEVICE_TOKEN_PATH
LOGOUT_URL = API_URL + oauth.LOGOUT_PATH


def _device_authorize_body() -> dict:
    return {
        "device_code": "dc",
        "user_code": "USER",
        "verification_uri": "https://login.baseten.co/device",
        "verification_uri_complete": "https://login.baseten.co/device?user_code=USER",
        "expires_in": 600,
        "interval": 0,
    }


def test_request_device_authorization_success():
    with requests_mock.Mocker() as m:
        m.post(DEVICE_AUTHORIZE_URL, json=_device_authorize_body())
        auth = oauth.request_device_authorization(API_URL)
    assert auth.device_code == "dc"
    assert auth.user_code == "USER"
    assert auth.verification_uri == "https://login.baseten.co/device"


def test_request_device_authorization_404_raises():
    with requests_mock.Mocker() as m:
        m.post(DEVICE_AUTHORIZE_URL, status_code=404, text="<html>not found</html>")
        with pytest.raises(oauth.OAuthError, match="404"):
            oauth.request_device_authorization(API_URL)


def test_poll_device_token_success():
    auth = oauth.DeviceAuthorization(
        device_code="dc",
        user_code="USER",
        verification_uri="https://login.baseten.co/device",
        verification_uri_complete=None,
        expires_in=600,
        interval=0,
    )
    with requests_mock.Mocker() as m:
        m.post(
            DEVICE_TOKEN_URL,
            json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        )
        cred = oauth.poll_device_token(API_URL, auth)
    assert cred.access_token == "at"
    assert cred.refresh_token == "rt"
    assert cred.expires_at > int(time.time())


def test_poll_device_token_authorization_pending_then_success():
    auth = oauth.DeviceAuthorization(
        device_code="dc",
        user_code="USER",
        verification_uri="https://login.baseten.co/device",
        verification_uri_complete=None,
        expires_in=600,
        interval=0,
    )
    with requests_mock.Mocker() as m:
        m.post(
            DEVICE_TOKEN_URL,
            [
                {"json": {"error": "authorization_pending"}, "status_code": 400},
                {
                    "json": {
                        "access_token": "at",
                        "refresh_token": "rt",
                        "expires_in": 3600,
                    },
                    "status_code": 200,
                },
            ],
        )
        cred = oauth.poll_device_token(API_URL, auth)
    assert cred.access_token == "at"


def test_poll_device_token_slow_down_increases_interval():
    auth = oauth.DeviceAuthorization(
        device_code="dc",
        user_code="USER",
        verification_uri="https://login.baseten.co/device",
        verification_uri_complete=None,
        expires_in=600,
        interval=0,
    )
    sleeps: list[float] = []
    with requests_mock.Mocker() as m, pytest.MonkeyPatch.context() as mp:
        mp.setattr(oauth.time, "sleep", lambda s: sleeps.append(s))
        m.post(
            DEVICE_TOKEN_URL,
            [
                {"json": {"error": "slow_down"}, "status_code": 400},
                {
                    "json": {
                        "access_token": "at",
                        "refresh_token": "rt",
                        "expires_in": 3600,
                    },
                    "status_code": 200,
                },
            ],
        )
        oauth.poll_device_token(API_URL, auth)
    assert sleeps and sleeps[0] >= 5


def test_poll_device_token_expired_token_raises():
    auth = oauth.DeviceAuthorization(
        device_code="dc",
        user_code="USER",
        verification_uri="https://login.baseten.co/device",
        verification_uri_complete=None,
        expires_in=600,
        interval=0,
    )
    with requests_mock.Mocker() as m:
        m.post(DEVICE_TOKEN_URL, json={"error": "expired_token"}, status_code=400)
        with pytest.raises(oauth.OAuthError, match="expired_token"):
            oauth.poll_device_token(API_URL, auth)


def test_poll_device_token_access_denied_raises():
    auth = oauth.DeviceAuthorization(
        device_code="dc",
        user_code="USER",
        verification_uri="https://login.baseten.co/device",
        verification_uri_complete=None,
        expires_in=600,
        interval=0,
    )
    with requests_mock.Mocker() as m:
        m.post(
            DEVICE_TOKEN_URL,
            json={"error": "access_denied", "error_description": "user denied"},
            status_code=400,
        )
        with pytest.raises(oauth.OAuthError, match="access_denied"):
            oauth.poll_device_token(API_URL, auth)


def test_poll_device_token_deadline_raises(monkeypatch):
    auth = oauth.DeviceAuthorization(
        device_code="dc",
        user_code="USER",
        verification_uri="https://login.baseten.co/device",
        verification_uri_complete=None,
        expires_in=0,
        interval=0,
    )
    with pytest.raises(oauth.OAuthError, match="expired"):
        oauth.poll_device_token(API_URL, auth)


def test_refresh_success():
    cred = oauth.OAuthCredential(
        access_token="old", refresh_token="rt-old", expires_at=0
    )
    with requests_mock.Mocker() as m:
        m.post(
            DEVICE_TOKEN_URL,
            json={"access_token": "new", "refresh_token": "rt-new", "expires_in": 3600},
        )
        refreshed = oauth.refresh(API_URL, cred)
    assert refreshed.access_token == "new"
    assert refreshed.refresh_token == "rt-new"


def test_refresh_failure_raises():
    cred = oauth.OAuthCredential(
        access_token="old", refresh_token="rt-old", expires_at=0
    )
    with requests_mock.Mocker() as m:
        m.post(DEVICE_TOKEN_URL, status_code=401, text="invalid_grant")
        with pytest.raises(oauth.OAuthError, match="401"):
            oauth.refresh(API_URL, cred)


def test_revoke_swallows_errors(caplog):
    cred = oauth.OAuthCredential(access_token="at", refresh_token="rt", expires_at=0)
    with requests_mock.Mocker() as m:
        m.post(LOGOUT_URL, status_code=500, text="boom")
        oauth.revoke(API_URL, cred)
    assert any("revoke" in r.message.lower() for r in caplog.records)


def test_run_device_flow_end_to_end(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(
        oauth.webbrowser, "open", lambda url, new=0: opened.append(url) or True
    )
    with requests_mock.Mocker() as m:
        m.post(DEVICE_AUTHORIZE_URL, json=_device_authorize_body())
        m.post(
            DEVICE_TOKEN_URL,
            json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        )
        cred = oauth.run_device_flow(API_URL)
    assert cred.access_token == "at"
    assert opened == ["https://login.baseten.co/device?user_code=USER"]
    out = capsys.readouterr().out
    assert "https://login.baseten.co/device?user_code=USER" in out
    assert "USER" in out


def test_run_device_flow_browser_open_failure_still_prints(monkeypatch, capsys):
    def _raise(url, new=0):
        raise RuntimeError("no browser")

    monkeypatch.setattr(oauth.webbrowser, "open", _raise)
    with requests_mock.Mocker() as m:
        m.post(DEVICE_AUTHORIZE_URL, json=_device_authorize_body())
        m.post(
            DEVICE_TOKEN_URL,
            json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        )
        cred = oauth.run_device_flow(API_URL)
    assert cred.access_token == "at"
    out = capsys.readouterr().out
    assert "https://login.baseten.co/device?user_code=USER" in out
    assert "USER" in out


TOKEN_EXCHANGE_URL = API_URL + oauth.TOKEN_EXCHANGE_PATH


def test_exchange_identity_token_request_shape():
    with requests_mock.Mocker() as m:
        m.post(
            TOKEN_EXCHANGE_URL,
            json={"access_token": "at", "token_type": "Bearer", "expires_in": 3600},
        )
        token = oauth.exchange_identity_token(API_URL, "idp-jwt", "rule-1", "/token")
    assert m.last_request.json() == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": "idp-jwt",
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "audience": "rule-1",
    }
    assert m.last_request.timeout == oauth.EXCHANGE_TIMEOUT_SECONDS
    assert token.access_token == "at"
    assert token.expires_at > int(time.time()) + 3500
    assert token.lifetime_seconds == 3600


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"token_type": "Bearer", "expires_in": 3600}, "no access_token"),
        (
            {"access_token": "at", "token_type": "mac", "expires_in": 3600},
            "unsupported token_type",
        ),
        (
            {"access_token": "at", "token_type": "Bearer", "expires_in": 0},
            "invalid expires_in",
        ),
    ],
)
def test_exchange_identity_token_rejects_bad_response(payload, message):
    with requests_mock.Mocker() as m:
        m.post(TOKEN_EXCHANGE_URL, json=payload)
        with pytest.raises(oauth.OAuthError, match=message):
            oauth.exchange_identity_token(API_URL, "idp-jwt", "rule-1", "/token")


def test_exchange_identity_token_invalid_grant():
    with requests_mock.Mocker() as m:
        m.post(TOKEN_EXCHANGE_URL, status_code=400, json={"error": "invalid_grant"})
        with pytest.raises(
            oauth.OAuthError, match="refused.*rule-1.*/token.*400.*invalid_grant"
        ):
            oauth.exchange_identity_token(API_URL, "idp-jwt", "rule-1", "/token")


def test_exchange_identity_token_does_not_follow_redirects():
    with requests_mock.Mocker() as m:
        m.post(
            TOKEN_EXCHANGE_URL,
            status_code=307,
            headers={"Location": "https://elsewhere.example/token"},
        )
        elsewhere = m.post("https://elsewhere.example/token", json={})
        with pytest.raises(oauth.OAuthError, match="307"):
            oauth.exchange_identity_token(API_URL, "idp-jwt", "rule-1", "/token")
    assert not elsewhere.called


def test_exchange_identity_token_other_error():
    with requests_mock.Mocker() as m:
        m.post(TOKEN_EXCHANGE_URL, status_code=503, text="unavailable")
        with pytest.raises(oauth.OAuthError, match="503"):
            oauth.exchange_identity_token(API_URL, "idp-jwt", "rule-1", "/token")


def test_oauth_credential_keeps_the_full_leeway(monkeypatch):
    credential = oauth.OAuthCredential.from_token_response(
        {"access_token": "t", "refresh_token": "r", "expires_in": 90}
    )
    assert credential.lifetime_seconds is None
    now = time.time()
    monkeypatch.setattr(oauth.time, "time", lambda: now + 31)
    assert credential.is_expired()


def test_is_expired_accepts_a_leeway(monkeypatch):
    credential = oauth.OAuthCredential(
        access_token="t", refresh_token="r", expires_at=int(time.time()) + 90
    )
    assert not credential.is_expired(leeway=0)
    assert credential.is_expired(leeway=120)


def test_exchange_errors_are_workload_identity_errors():
    with requests_mock.Mocker() as m:
        m.post(TOKEN_EXCHANGE_URL, status_code=503, text="unavailable")
        with pytest.raises(oauth.WorkloadIdentityError):
            oauth.exchange_identity_token(API_URL, "idp-jwt", "rule-1", "/token")
