class Error(Exception):
    """Base Baseten Error"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ApiError(Error):
    """Errors in calling the Baseten API."""

    pass


class AuthorizationError(Error):
    """Raised in places where the user needs to be logged in and is not."""

    pass
