class Error(Exception):
    """Base Baseten Error"""
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class FrameworkNotSupportedError(Error):
    """Raised in places where the user attempts to use Baseten with an unsupported framework"""
    pass


class ModelFilesMissingError(Error):
    pass


class ModelClassImplementationError(Error):
    pass


class InvalidConfigurationError(Error):
    pass


class ValidationError(Error):
    pass
