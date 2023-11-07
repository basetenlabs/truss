class Error(Exception):
    """Base Baseten Error"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ApiError(Error):
    """Errors in calling the Baseten API."""

    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response

    def __str__(self):
        return self.message


class AuthorizationError(Error):
    """Raised in places where the user needs to be logged in and is not."""

    pass
