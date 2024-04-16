import logging
import os
from typing import Optional

from truss.remote.baseten.error import AuthorizationError

logger = logging.getLogger(__name__)


class ApiKey:
    value: str

    def __init__(self, value: str) -> None:
        self.value = value

    def header(self):
        return {"Authorization": f"Api-Key {self.value}"}


class AuthService:
    def __init__(self, api_key: Optional[str] = None) -> None:
        if not api_key:
            api_key = os.environ.get("BASETEN_API_KEY", None)
        self._api_key = api_key

    def validate(self) -> None:
        if not self._api_key:
            raise AuthorizationError("No API key provided.")

    def authenticate(self) -> ApiKey:
        self.validate()
        return ApiKey(self._api_key)  # type: ignore
