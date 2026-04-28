import logging
import os
from dataclasses import replace
from typing import Callable, Dict, Optional

from truss.remote.baseten.error import AuthorizationError
from truss.remote.baseten.oauth import OAuthCredential, refresh

logger = logging.getLogger(__name__)


class AuthService:
    """Holds the active credential for a remote and yields request headers.

    Two credential modes:
    - API key (legacy): ``AuthService(api_key="...")`` or ``BASETEN_API_KEY``.
    - OAuth: ``AuthService(api_url=..., oauth_credential=...)``. Access
      token is refreshed in-process via ``oauth.refresh`` when within the
      leeway of ``expires_at``; refreshed credential is handed to
      ``on_token_refresh`` so the caller can persist it.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        api_url: Optional[str] = None,
        oauth_credential: Optional[OAuthCredential] = None,
        on_token_refresh: Optional[Callable[[OAuthCredential], None]] = None,
    ) -> None:
        self._oauth_credential: Optional[OAuthCredential] = oauth_credential
        self._api_url: Optional[str] = api_url
        self._on_token_refresh: Optional[Callable[[OAuthCredential], None]] = None
        self._api_key: Optional[str] = None
        if oauth_credential is not None:
            if api_url is None:
                raise ValueError("api_url is required for OAuth credentials")
            self._on_token_refresh = on_token_refresh
            return
        if not api_key:
            api_key = os.environ.get("BASETEN_API_KEY")
        self._api_key = api_key

    def auth_header(self) -> Dict[str, str]:
        """Return a fresh ``Authorization`` header for any credential type.

        Refreshes the OAuth access token in-place when within the leeway of
        its ``expires_at``.
        """
        if self._oauth_credential is not None:
            self._refresh_if_expired()
            return {"Authorization": f"Bearer {self._oauth_credential.access_token}"}
        if not self._api_key:
            raise AuthorizationError("No credentials provided.")
        return {"Authorization": f"Api-Key {self._api_key}"}

    def _refresh_if_expired(self) -> None:
        assert self._oauth_credential is not None
        if not self._oauth_credential.is_expired():
            return
        assert self._api_url is not None
        refreshed = refresh(self._api_url, self._oauth_credential)
        self._oauth_credential = refreshed
        if self._on_token_refresh is not None:
            try:
                self._on_token_refresh(replace(refreshed))
            except Exception as exc:
                logger.warning("Persisting refreshed OAuth token failed: %s", exc)
