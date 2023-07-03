import logging
import os
from typing import Optional

from truss.remote.baseten.error import AuthorizationError

logger = logging.getLogger(__name__)


class ApiKey:
    value: str

    def __init__(self, value: str):
        self.value = value

    def header(self):
        return {"Authorization": f"Api-Key {self.value}"}


class AuthService:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is not None:
            self.set_key(api_key)

    def validate(self):
        if "BASETEN_API_KEY" not in os.environ:
            raise AuthorizationError(
                "Could not find BASETEN_API_KEY in environment variables."
            )

    def authenticate(self) -> ApiKey:
        self.validate()
        return ApiKey(os.environ["BASETEN_API_KEY"])

    def set_key(self, api_key: str) -> ApiKey:
        logger.warning(
            "Setting BASETEN_API_KEY in environment variables. This may not persist."
        )
        os.environ["BASETEN_API_KEY"] = api_key
        return self.authenticate()
