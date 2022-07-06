try:
    import kfserving  # noqa: F401
    KFSERVING_LIB = True
except ModuleNotFoundError:
    KFSERVING_LIB = False


def ensure_kfserving_installed():
    if not KFSERVING_LIB:
        raise ModuleNotFoundError('Could not successfully import "kfserving" package, check your Python environment')
    return True
