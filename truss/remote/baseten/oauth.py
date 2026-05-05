"""OAuth 2.0 device authorization grant (RFC 8628) for the Baseten backend.

Tokens are returned as :class:`OAuthCredential` with an absolute Unix
``expires_at`` so callers can do proactive refresh before sending.
"""

import logging
import time
from typing import Optional

import pydantic
import requests

from truss.remote.baseten.user_agent import user_agent_header

logger = logging.getLogger(__name__)

CLIENT_ID = "baseten-cli"
DEVICE_AUTHORIZE_PATH = "/v1/users/auth/device/authorize"
DEVICE_TOKEN_PATH = "/v1/users/auth/device/token"
LOGOUT_PATH = "/v1/users/auth/logout"

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
REFRESH_GRANT_TYPE = "refresh_token"

DEFAULT_POLL_INTERVAL_SECONDS = 5
EXPIRY_LEEWAY_SECONDS = 60


class OAuthError(Exception):
    """Raised on terminal OAuth failures (denied, expired, invalid request)."""


class OAuthCredential(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    access_token: str = pydantic.Field(min_length=1)
    refresh_token: str = pydantic.Field(min_length=1)
    expires_at: int  # absolute Unix timestamp seconds

    def is_expired(self, leeway: int = EXPIRY_LEEWAY_SECONDS) -> bool:
        return time.time() + leeway >= self.expires_at

    @classmethod
    def from_token_response(cls, payload: dict) -> "OAuthCredential":
        expires_in = payload.get("expires_in")
        if expires_in is None:
            raise OAuthError(
                f"Token response missing required fields: {sorted(payload)}"
            )
        payload = {**payload, "expires_at": int(time.time()) + int(expires_in)}
        try:
            return cls.model_validate(payload)
        except pydantic.ValidationError as exc:
            raise OAuthError(f"Token response missing required fields: {exc}") from exc


class DeviceAuthorization(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str] = None
    expires_in: int = 600
    interval: int = DEFAULT_POLL_INTERVAL_SECONDS


def request_device_authorization(api_url: str) -> DeviceAuthorization:
    resp = requests.post(
        api_url.rstrip("/") + DEVICE_AUTHORIZE_PATH,
        data={"client_id": CLIENT_ID},
        headers={"User-Agent": user_agent_header()},
        timeout=30,
    )
    if not resp.ok:
        raise OAuthError(
            f"Device authorize failed ({resp.status_code}): {resp.text.strip()}"
        )
    try:
        return DeviceAuthorization.model_validate(resp.json())
    except pydantic.ValidationError as exc:
        raise OAuthError(f"Device authorize response malformed: {exc}") from exc


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
            headers={"User-Agent": user_agent_header()},
            timeout=30,
        )
        if resp.ok:
            return OAuthCredential.from_token_response(resp.json())
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
        headers={"User-Agent": user_agent_header()},
        timeout=30,
    )
    if not resp.ok:
        raise OAuthError(
            f"Token refresh failed ({resp.status_code}): {resp.text.strip()}"
        )
    return OAuthCredential.from_token_response(resp.json())


def revoke(api_url: str, credential: OAuthCredential) -> None:
    """Best-effort logout. Logs and swallows non-2xx so cleanup proceeds."""
    try:
        resp = requests.post(
            api_url.rstrip("/") + LOGOUT_PATH,
            headers={
                "Authorization": f"Bearer {credential.access_token}",
                "User-Agent": user_agent_header(),
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.warning("Token revoke request failed: %s", exc)
        return
    if not resp.ok:
        logger.warning(
            "Token revoke returned %s: %s", resp.status_code, resp.text.strip()
        )
