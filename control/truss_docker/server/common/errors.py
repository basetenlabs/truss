class Error(Exception):
    """Base Error"""

    def __init__(self, message):
        super(Error, self).__init__(message)
        self.message = message


class InferenceError(Error):
    """Error raised when model inference fails."""
