def proxy_to_shadow(func):
    def wrapper(*args, **kwargs):
        from truss.truss_handle import TrussHandle

        truss_handle = args[0]
        gathered_truss_handle = TrussHandle(truss_handle.gather())
        return func(gathered_truss_handle, *args[1:], **kwargs)

    return wrapper
