import logging
import threading
from typing import Callable, Optional, Union

import pydantic

from truss.remote.baseten.oauth import OAuthCredential, refresh

logger = logging.getLogger(__name__)


class ApiKeyCredential(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    api_key: str = pydantic.Field(min_length=1)


class OAuthSession(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    api_url: str
    credential: OAuthCredential
    on_token_refresh: Optional[Callable[[OAuthCredential], None]] = None


class AuthService:
    """Holds the active credential for a remote and yields request headers.

    OAuth access tokens are refreshed in-process via ``oauth.refresh`` when
    within the leeway of ``expires_at``; refreshed credential is handed to
    ``OAuthSession.on_token_refresh`` so the caller can persist it.
    """

    def __init__(self, credential: Union[ApiKeyCredential, OAuthSession]) -> None:
        self._credential = credential
        self._refresh_lock = threading.Lock()

    def fetch_auth_header(self) -> dict[str, str]:
        """Return a fresh ``Authorization`` header for any credential type.

        Refreshes the OAuth access token in-place when within the leeway of
        its ``expires_at``.
        """
        if isinstance(self._credential, OAuthSession):
            with self._refresh_lock:
                session = self._refresh_if_expired(self._credential)
            return {"Authorization": f"Bearer {session.credential.access_token}"}
        return {"Authorization": f"Api-Key {self._credential.api_key}"}

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
