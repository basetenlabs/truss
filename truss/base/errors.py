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


class ContainerIsDownError(Error):
    pass


class ContainerNotFoundError(Error):
    pass


class ContainerAPINoResponseError(Error):
    pass


class RemoteNetworkError(Exception):
    pass


class TrussUsageError(TypeError):
    """Raised when user-defined Chainlets do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""
