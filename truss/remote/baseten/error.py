from typing import Optional


class Error(Exception):
    """Base Baseten Error"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ApiError(Error):
    """Errors in calling the Baseten API."""

    def __init__(self, message: str, graphql_error_code: Optional[str] = None):
        super().__init__(message)
        self.graphql_error_code = graphql_error_code


class AuthorizationError(Error):
    """Raised in places where the user needs to be logged in and is not."""

    pass
