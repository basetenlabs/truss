class Error(Exception):
    """Base Truss Error"""

    def __init__(self, message: str):
        super(Error, self).__init__(message)
        self.message = message
        self.type = type


class UnsupportedTypeError(Error):
    """Raised when a Pydantic field type is not supported"""

    pass


class MissingFieldError(Error):
    """Raised when a Pydantic field is missing"""

    pass


class MissingInputClassError(Error):
    """Raised when the user does not define an input class"""

    pass


class MissingOutputClassError(Error):
    """Raised when the user does not define an output class"""

    pass


class InvalidModelResponseError(Error):
    """Raised when the model returns an invalid response"""

    pass
