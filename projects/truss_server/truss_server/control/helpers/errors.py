from truss_common.patch.errors import Error


class ModelNotReady(Error):
    """Model has started running, but not ready yet."""

    pass


class ModelLoadFailed(Error):
    """Model has failed to load."""

    pass
