class Error(Exception):
    """Base Truss Error"""

    def __init__(self, message: str):
        super(Error, self).__init__(message)
        self.message = message
        self.type = type


class PatchApplicatonError(Error):
    pass


class UnsupportedPatch(PatchApplicatonError):
    """Patch unsupported by this truss"""

    pass


class PatchFailedRecoverable(PatchApplicatonError):
    """Patch admissible but failed to apply. Recoverable via further patching."""

    pass


class PatchFailedUnrecoverable(PatchApplicatonError):
    """Patch admissible but failed to apply, leaving truss in unrecoverable state.
    Full deploy is required."""

    pass


class InadmissiblePatch(PatchApplicatonError):
    """Patch does not apply to current state of Truss."""

    pass


class ModelNotReady(Error):
    """Model has started running, but not ready yet."""

    pass
