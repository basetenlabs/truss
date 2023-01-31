class Error(Exception):
    """Base Truss Error"""

    def __init__(self, message: str):
        super(Error, self).__init__(message)
        self.message = message
        self.type = type


class UnsupportedPatch(Error):
    """Patch unsupported by this truss"""

    pass


class PatchFailedRecoverable(Error):
    """Patch admissible but failed to apply. Recoverable via further patching."""

    pass


class PatchFailedUnrecoverable(Error):
    """Patch admissible but failed to apply, leaving truss in unrecoverable state.
    Full deploy is required."""

    pass


class InadmissiblePatch(Error):
    """Patch does not apply to current state of Truss."""

    pass
