import functools
import os

from truss.remote.baseten.error import AuthorizationError


class AuthToken:
    value: str

    def __init__(self, value: str):
        self.value = value

    def headers(self):
        raise NotImplementedError


class ApiKey(AuthToken):
    def headers(self):
        return {"Authorization": f"Api-Key {self.value}"}


class JWT(AuthToken):
    def headers(self):
        return {"Authorization": self.value}


class AuthService:
    def validate(self):
        if "BASETEN_API_KEY" not in os.environ:
            raise AuthorizationError(
                "Could not find BASETEN_API_KEY in environment variables."
            )

    def authenticate(self) -> AuthToken:
        self.validate()
        return ApiKey(os.environ["BASETEN_API_KEY"])

    def set_key(self, api_key: str) -> AuthToken:
        os.environ["BASETEN_API_KEY"] = api_key
        return self.authenticate()


def with_api_key(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        baseten_api_key = AuthService().authenticate()

        result = func(baseten_api_key, *args, **kwargs)

        return result

    return wrapper
