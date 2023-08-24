import inspect


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    """Checks if a signature accepts **kwargs."""
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    """Checks if a signature accepts a keyword argument with the given name."""
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)
