import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pydantic

from truss.remote.baseten.oauth import (
    AccessToken,
    OAuthCredential,
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


BASETEN_API_KEY_ENV = "BASETEN_API_KEY"
IDENTITY_TOKEN_FILE_ENV = "BASETEN_IDENTITY_TOKEN_FILE"
TRUST_RULE_ID_ENV = "BASETEN_TRUST_RULE_ID"


def workload_identity_from_env() -> Optional[Tuple[Path, str]]:
    """Return ``(identity_token_file, trust_rule_id)`` when workload identity
    federation is configured in the environment, or None. Setting just one of
    the two variables is an error."""
    token_file = os.environ.get(IDENTITY_TOKEN_FILE_ENV)
    trust_rule_id = os.environ.get(TRUST_RULE_ID_ENV)
    if not token_file and not trust_rule_id:
        return None
    if not token_file or not trust_rule_id:
        missing, present = (
            (IDENTITY_TOKEN_FILE_ENV, TRUST_RULE_ID_ENV)
            if not token_file
            else (TRUST_RULE_ID_ENV, IDENTITY_TOKEN_FILE_ENV)
        )
        raise ValueError(
            f"{present} is set but {missing} is not; workload identity "
            "federation needs both."
        )
    return Path(token_file).expanduser(), trust_rule_id


class WorkloadIdentityCredential(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    api_url: str
    identity_token_file: Path
    trust_rule_id: str = pydantic.Field(min_length=1)


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
        if token is not None and not token.is_expired():
            return token
        token = exchange_identity_token(
            credential.api_url, credential.identity_token_file, credential.trust_rule_id
        )
        self._workload_identity_token = token
        return token
