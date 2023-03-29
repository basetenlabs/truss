class Error(Exception):
    """Base Error"""

    def __init__(self, message):
        super(Error, self).__init__(message)
        self.message = message


class InferenceError(Error):
    """Error raised when model inference fails."""


class ModelMissingError(Error):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class ModelNotReady(Error):
    def __init__(self, model_name: str, detail: str = None):
        self.model_name = model_name
        self.error_msg = f"Model with name {self.model_name} is not ready."
        if detail:
            self.error_msg = self.error_msg + " " + detail

    def __str__(self):
        return self.error_msg
