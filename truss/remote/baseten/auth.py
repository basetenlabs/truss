import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

import pydantic
import requests

from truss.remote.baseten.oauth import (
    AccessToken,
    OAuthCredential,
    OAuthError,
    WorkloadIdentityError,
    exchange_identity_token,
    refresh,
)

logger = logging.getLogger(__name__)


class ApiKeyCredential(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    api_key: str = pydantic.Field(min_length=1)
    # Send as `Bearer` rather than the legacy `Api-Key`. Set for env-supplied
    # tokens, which may be an API key or an OAuth access token; trussrc remotes
    # keep the legacy scheme.
    use_bearer: bool = False


class OAuthSession(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    api_url: str
    credential: OAuthCredential
    on_token_refresh: Optional[Callable[[OAuthCredential], None]] = None


# How long before expiry a workload identity access token is replaced, capped
# at half its lifetime; as in the Baseten SDKs.
WORKLOAD_IDENTITY_REFRESH_LEEWAY_SECONDS = 300

BASETEN_API_KEY_ENV = "BASETEN_API_KEY"
# Set by the baseten CLI when it hands its credential to truss.
TRUSS_AUTH_API_KEY_ENV = "BASETEN_TRUSS_AUTH_API_KEY"
IDENTITY_TOKEN_FILE_ENV = "BASETEN_WORKLOAD_IDENTITY_TOKEN_FILE"
IDENTITY_TOKEN_ENV = "BASETEN_WORKLOAD_IDENTITY_TOKEN"
TRUST_RULE_ID_ENV = "BASETEN_WORKLOAD_IDENTITY_TRUST_RULE_ID"


class WorkloadIdentityConfig(pydantic.BaseModel):
    """Workload identity federation settings from the environment. Exactly one
    of ``identity_token_file`` and ``identity_token`` is set."""

    model_config = pydantic.ConfigDict(frozen=True)

    trust_rule_id: str = pydantic.Field(min_length=1)
    identity_token_file: Optional[Path] = None
    # Unlike a file, it cannot be replaced while truss runs, so exchanges fail
    # once it expires.
    identity_token: Optional[pydantic.SecretStr] = None

    @pydantic.model_validator(mode="after")
    def _one_identity_token(self) -> "WorkloadIdentityConfig":
        if (self.identity_token_file is None) == (self.identity_token is None):
            raise ValueError(
                "Exactly one of identity_token_file and identity_token is required."
            )
        return self

    @property
    def identity_token_source(self) -> str:
        """Where the identity token comes from, for messages."""
        if self.identity_token is not None:
            return IDENTITY_TOKEN_ENV
        return str(self.identity_token_file)


def workload_identity_from_env() -> Optional[WorkloadIdentityConfig]:
    """Return the workload identity federation settings, or None when none of
    its variables is set.

    Incomplete settings, both identity token variables, or an API key variable
    alongside them are errors: picking one silently could run as a different
    identity.
    """
    token_file = os.environ.get(IDENTITY_TOKEN_FILE_ENV)
    token = os.environ.get(IDENTITY_TOKEN_ENV)
    trust_rule_id = os.environ.get(TRUST_RULE_ID_ENV)
    if not token_file and not token and not trust_rule_id:
        return None
    if token_file and token:
        raise ValueError(
            f"Both {IDENTITY_TOKEN_FILE_ENV} and {IDENTITY_TOKEN_ENV} are set; "
            "set only one."
        )
    if not trust_rule_id:
        present = IDENTITY_TOKEN_FILE_ENV if token_file else IDENTITY_TOKEN_ENV
        raise ValueError(
            f"{present} is set but {TRUST_RULE_ID_ENV} is not; workload identity "
            "federation needs both."
        )
    if not token_file and not token:
        raise ValueError(
            f"{TRUST_RULE_ID_ENV} is set but neither {IDENTITY_TOKEN_FILE_ENV} nor "
            f"{IDENTITY_TOKEN_ENV} is; workload identity federation needs an "
            "identity token."
        )
    for api_key_env in (BASETEN_API_KEY_ENV, TRUSS_AUTH_API_KEY_ENV):
        if os.environ.get(api_key_env):
            raise ValueError(
                f"Both {api_key_env} and workload identity ({TRUST_RULE_ID_ENV}) "
                "are set; unset one."
            )
    return WorkloadIdentityConfig(
        trust_rule_id=trust_rule_id,
        identity_token_file=Path(token_file).expanduser() if token_file else None,
        identity_token=pydantic.SecretStr(token) if token else None,
    )


class WorkloadIdentityCredential(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    api_url: str
    config: WorkloadIdentityConfig


class AuthService:
    """Holds the active credential for a remote and yields request headers.

    OAuth access tokens are refreshed in-process via ``oauth.refresh`` when
    within the leeway of ``expires_at``; refreshed credential is handed to
    ``OAuthSession.on_token_refresh`` so the caller can persist it.

    Workload identity access tokens are exchanged the same way and kept in
    memory only.
    """

    def __init__(
        self,
        credential: Union[ApiKeyCredential, OAuthSession, WorkloadIdentityCredential],
    ) -> None:
        self._credential = credential
        self._refresh_lock = threading.Lock()
        self._workload_identity_token: Optional[AccessToken] = None

    def fetch_auth_header(self) -> dict[str, str]:
        """Return a fresh ``Authorization`` header for any credential type.

        Refreshes the OAuth access token in-place when within the leeway of
        its ``expires_at``.
        """
        if isinstance(self._credential, OAuthSession):
            with self._refresh_lock:
                session = self._refresh_if_expired(self._credential)
            return {"Authorization": f"Bearer {session.credential.access_token}"}
        if isinstance(self._credential, WorkloadIdentityCredential):
            with self._refresh_lock:
                token = self._exchange_if_expired(self._credential)
            return {"Authorization": f"Bearer {token.access_token}"}
        scheme = "Bearer" if self._credential.use_bearer else "Api-Key"
        return {"Authorization": f"{scheme} {self._credential.api_key}"}

    def note_response(self, response: requests.Response) -> None:
        """Drop a workload identity access token Baseten refused before its
        expiry, for example after revocation, so the next request exchanges for
        a new one. The refused request is not sent again, since it may have side
        effects."""
        if not isinstance(
            self._credential, WorkloadIdentityCredential
        ) or not refused_by_baseten(response):
            return
        sent = (
            response.request.headers.get("Authorization") if response.request else None
        )
        with self._refresh_lock:
            token = self._workload_identity_token
            # Compared by value, so a refusal of an older token never discards
            # a newer one.
            if token is not None and sent == f"Bearer {token.access_token}":
                self._workload_identity_token = None

    def _refresh_if_expired(self, session: OAuthSession) -> OAuthSession:
        if not session.credential.is_expired():
            return session
        refreshed = refresh(session.api_url, session.credential)
        updated = session.model_copy(update={"credential": refreshed})
        self._credential = updated
        if session.on_token_refresh is not None:
            try:
                session.on_token_refresh(refreshed.model_copy())
            except Exception as exc:
                logger.warning("Persisting refreshed OAuth token failed: %s", exc)
        return updated

    def _exchange_if_expired(
        self, credential: WorkloadIdentityCredential
    ) -> AccessToken:
        token = self._workload_identity_token
        if token is not None and time.time() < _refresh_at(token):
            return token
        try:
            config = credential.config
            token = exchange_identity_token(
                credential.api_url,
                _read_identity_token(config),
                config.trust_rule_id,
                config.identity_token_source,
            )
        except (OAuthError, requests.RequestException) as exc:
            # Refreshes start before expiry, so a failed one can keep using the
            # current token until it actually expires.
            previous = self._workload_identity_token
            if previous is None or time.time() >= previous.expires_at:
                raise
            logger.warning(
                "Refreshing the workload identity access token failed; using "
                "the current one until it expires: %s",
                exc,
            )
            return previous
        self._workload_identity_token = token
        return token


def _read_identity_token(config: WorkloadIdentityConfig) -> str:
    """Return the identity token, re-reading the file on every exchange because
    platforms such as Kubernetes rotate it in place."""
    if config.identity_token is not None:
        token = config.identity_token.get_secret_value().strip()
        if not token:
            raise WorkloadIdentityError(f"{IDENTITY_TOKEN_ENV} is empty.")
        return token
    assert config.identity_token_file is not None
    try:
        token = config.identity_token_file.read_text().strip()
    except OSError as exc:
        raise WorkloadIdentityError(
            f"Cannot read the identity token file {config.identity_token_file}: {exc}"
        ) from exc
    if not token:
        raise WorkloadIdentityError(
            f"The identity token file {config.identity_token_file} is empty."
        )
    return token


def _refresh_at(token: AccessToken) -> float:
    lifetime = token.lifetime_seconds or 0
    return token.expires_at - min(
        WORKLOAD_IDENTITY_REFRESH_LEEWAY_SECONDS, lifetime / 2
    )


def refused_by_baseten(response: requests.Response) -> bool:
    """Report whether Baseten's auth layer refused the request's access token.

    It signals that with this challenge. A model's own error response cannot
    carry it, since its headers are not forwarded.
    """
    if response.status_code != 401:
        return False
    challenge = response.headers.get("WWW-Authenticate", "")
    # requests joins repeated headers with ", ", so check each challenge.
    return any(
        part.strip().startswith("Bearer ")
        and 'realm="baseten"' in part
        and 'error="invalid_token"' in part
        for part in re.split(r",\s*(?=Bearer )", challenge)
    )
