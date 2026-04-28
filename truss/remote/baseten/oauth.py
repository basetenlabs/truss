"""OAuth 2.0 device authorization grant (RFC 8628) for the Baseten backend.

Tokens are returned as :class:`OAuthCredential` with an absolute Unix
``expires_at`` so callers can do proactive refresh before sending.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CLIENT_ID = "baseten-cli"
DEVICE_AUTHORIZE_PATH = "/v1/users/auth/device/authorize"
DEVICE_TOKEN_PATH = "/v1/users/auth/device/token"
LOGOUT_PATH = "/v1/users/auth/logout"

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
REFRESH_GRANT_TYPE = "refresh_token"

DEFAULT_POLL_INTERVAL_SECONDS = 5
EXPIRY_LEEWAY_SECONDS = 60


@dataclass(frozen=True)
class OAuthCredential:
    access_token: str
    refresh_token: str
    expires_at: int  # absolute Unix timestamp seconds

    def is_expired(self, leeway: int = EXPIRY_LEEWAY_SECONDS) -> bool:
        return time.time() + leeway >= self.expires_at


@dataclass(frozen=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str]
    expires_in: int
    interval: int


class OAuthError(Exception):
    """Raised on terminal OAuth failures (denied, expired, invalid request)."""


def _token_from_response(payload: dict) -> OAuthCredential:
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not access_token or not refresh_token or expires_in is None:
        raise OAuthError(f"Token response missing required fields: {sorted(payload)}")
    return OAuthCredential(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=int(time.time()) + int(expires_in),
    )


def request_device_authorization(api_url: str) -> DeviceAuthorization:
    resp = requests.post(
        api_url.rstrip("/") + DEVICE_AUTHORIZE_PATH,
        data={"client_id": CLIENT_ID},
        timeout=30,
    )
    if not resp.ok:
        raise OAuthError(
            f"Device authorize failed ({resp.status_code}): {resp.text.strip()}"
        )
    body = resp.json()
    return DeviceAuthorization(
        device_code=body["device_code"],
        user_code=body["user_code"],
        verification_uri=body["verification_uri"],
        verification_uri_complete=body.get("verification_uri_complete"),
        expires_in=int(body.get("expires_in", 600)),
        interval=int(body.get("interval", DEFAULT_POLL_INTERVAL_SECONDS)),
    )


def poll_device_token(
    api_url: str, authorization: DeviceAuthorization
) -> OAuthCredential:
    """Poll the token endpoint until the user authorizes or the grant fails.

    Implements RFC 8628 polling: ``authorization_pending`` → keep polling,
    ``slow_down`` → bump interval, terminal errors → raise.
    """
    token_url = api_url.rstrip("/") + DEVICE_TOKEN_PATH
    interval = max(1, authorization.interval)
    deadline = time.time() + authorization.expires_in
    while True:
        if time.time() >= deadline:
            raise OAuthError("Device authorization expired before completion.")
        resp = requests.post(
            token_url,
            data={
                "grant_type": DEVICE_GRANT_TYPE,
                "device_code": authorization.device_code,
                "client_id": CLIENT_ID,
            },
            timeout=30,
        )
        if resp.ok:
            return _token_from_response(resp.json())
        try:
            err = resp.json()
        except ValueError:
            raise OAuthError(
                f"Device token endpoint returned {resp.status_code}: "
                f"{resp.text.strip()}"
            )
        code = err.get("error")
        if code == "authorization_pending":
            time.sleep(interval)
            continue
        if code == "slow_down":
            interval += 5
            time.sleep(interval)
            continue
        raise OAuthError(
            f"Device authorization failed: {code or resp.status_code} "
            f"{err.get('error_description', '')}".strip()
        )


def run_device_flow(api_url: str) -> OAuthCredential:
    """Drive the full device flow: authorize, prompt, poll, return credential."""
    authorization = request_device_authorization(api_url)
    logger.info(
        "Enter code %s at %s", authorization.user_code, authorization.verification_uri
    )
    return poll_device_token(api_url, authorization)


def refresh(api_url: str, credential: OAuthCredential) -> OAuthCredential:
    resp = requests.post(
        api_url.rstrip("/") + DEVICE_TOKEN_PATH,
        data={
            "grant_type": REFRESH_GRANT_TYPE,
            "refresh_token": credential.refresh_token,
            "client_id": CLIENT_ID,
        },
        timeout=30,
    )
    if not resp.ok:
        raise OAuthError(
            f"Token refresh failed ({resp.status_code}): {resp.text.strip()}"
        )
    return _token_from_response(resp.json())


def revoke(api_url: str, credential: OAuthCredential) -> None:
    """Best-effort logout. Logs and swallows non-2xx so cleanup proceeds."""
    try:
        resp = requests.post(
            api_url.rstrip("/") + LOGOUT_PATH,
            headers={"Authorization": f"Bearer {credential.access_token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.warning("Token revoke request failed: %s", exc)
        return
    if not resp.ok:
        logger.warning(
            "Token revoke returned %s: %s", resp.status_code, resp.text.strip()
        )
