import functools
import os
from typing import Optional

from truss.remote.baseten.error import AuthorizationError


class ApiKey:
    value: str

    def __init__(self, value: str):
        self.value = value

    def headers(self):
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
        os.environ["BASETEN_API_KEY"] = api_key
        return self.authenticate()


def with_api_key(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        baseten_api_key = AuthService().authenticate()

        result = func(baseten_api_key, *args, **kwargs)

        return result

    return wrapper
