"""OAuth 2.0 device authorization grant (RFC 8628) for the Baseten backend.

Tokens are returned as :class:`OAuthCredential` with an absolute Unix
``expires_at`` so callers can do proactive refresh before sending.
"""

import logging
import time
import webbrowser
from pathlib import Path
from typing import Optional, Type, TypeVar

import pydantic
import requests

from truss.cli.utils.output import console
from truss.remote.baseten.user_agent import user_agent_header

logger = logging.getLogger(__name__)

CLIENT_ID = "baseten-cli"
DEVICE_AUTHORIZE_PATH = "/v1/users/auth/device/authorize"
DEVICE_TOKEN_PATH = "/v1/users/auth/device/token"
LOGOUT_PATH = "/v1/users/auth/logout"
TOKEN_EXCHANGE_PATH = "/v1/auth/token"

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
REFRESH_GRANT_TYPE = "refresh_token"

DEFAULT_POLL_INTERVAL_SECONDS = 5
EXPIRY_LEEWAY_SECONDS = 60


class OAuthError(Exception):
    """Raised on terminal OAuth failures (denied, expired, invalid request)."""


class AccessToken(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    access_token: str = pydantic.Field(min_length=1)
    expires_at: int  # absolute Unix timestamp seconds
    # Seconds the token was issued for; unknown for credentials loaded from storage.
    lifetime_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        leeway = EXPIRY_LEEWAY_SECONDS
        if self.lifetime_seconds is not None:
            # A token issued for a minute or less would otherwise be expired on arrival.
            leeway = min(leeway, self.lifetime_seconds // 2)
        return time.time() + leeway >= self.expires_at

    @classmethod
    def from_token_response(
        cls: Type["_AccessTokenT"], payload: dict
    ) -> "_AccessTokenT":
        expires_in = payload.get("expires_in")
        if expires_in is None:
            raise OAuthError(
                f"Token response missing required fields: {sorted(payload)}"
            )
        payload = {
            **payload,
            "expires_at": int(time.time()) + int(expires_in),
            "lifetime_seconds": int(expires_in),
        }
        try:
            return cls.model_validate(payload)
        except pydantic.ValidationError as exc:
            raise OAuthError(f"Token response missing required fields: {exc}") from exc


_AccessTokenT = TypeVar("_AccessTokenT", bound=AccessToken)


class OAuthCredential(AccessToken):
    refresh_token: str = pydantic.Field(min_length=1)


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
    """Drive the full device flow: authorize, open browser, poll, return credential."""
    authorization = request_device_authorization(api_url)
    target = authorization.verification_uri_complete or authorization.verification_uri
    try:
        webbrowser.open(target, new=2)
    except Exception:
        pass
    console.print("Browser opened to authenticate...")
    console.print()
    console.print("If it didn't open, visit:")
    console.print(f"  {target}")
    console.print()
    console.print(f"Verification code: {authorization.user_code}")
    console.print()
    console.print("Waiting...")
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


def exchange_identity_token(
    api_url: str, identity_token_file: Path, trust_rule_id: str
) -> AccessToken:
    """Trade a workload's identity provider JWT for a Baseten access token
    (workload identity federation, RFC 8693 token exchange with the trust rule
    as the audience)."""
    try:
        assertion = identity_token_file.read_text().strip()
    except OSError as exc:
        raise OAuthError(
            f"Cannot read the identity token file {identity_token_file}: {exc}"
        ) from exc
    if not assertion:
        raise OAuthError(f"The identity token file {identity_token_file} is empty.")
    resp = requests.post(
        api_url.rstrip("/") + TOKEN_EXCHANGE_PATH,
        json={
            "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
            "subject_token": assertion,
            "subject_token_type": JWT_TOKEN_TYPE,
            "audience": trust_rule_id,
        },
        headers={"User-Agent": user_agent_header()},
        timeout=30,
    )
    if resp.ok:
        return AccessToken.from_token_response(resp.json())
    try:
        code = resp.json().get("error")
    except ValueError:
        code = None
    # The server never says why it refused, so point at both inputs.
    if code == "invalid_grant":
        raise OAuthError(
            f"Baseten refused to exchange the identity token for trust rule "
            f"{trust_rule_id!r}. Check that the trust rule exists and that the "
            f"token in {identity_token_file} is current and matches the rule."
        )
    raise OAuthError(
        f"Identity token exchange failed ({resp.status_code}): {resp.text.strip()}"
    )


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
