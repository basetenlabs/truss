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


def test_run_device_flow_end_to_end():
    with requests_mock.Mocker() as m:
        m.post(DEVICE_AUTHORIZE_URL, json=_device_authorize_body())
        m.post(
            DEVICE_TOKEN_URL,
            json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        )
        cred = oauth.run_device_flow(API_URL)
    assert cred.access_token == "at"
