import json


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
        error_str = self.message
        if (
            self.response is not None
        ):  # non-200 Response objects are falsy, hence the not None.
            error_message = json.loads(self.response.content)
            error_message = (
                error_message["error"] if "error" in error_message else error_message
            )
            error_str = f"{error_str}\n<Server response: {error_message}>"
        return error_str


class AuthorizationError(Error):
    """Raised in places where the user needs to be logged in and is not."""

    pass
