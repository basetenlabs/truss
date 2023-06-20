def proxy_to_shadow(func):
    def wrapper(*args, **kwargs):
        from truss.truss_handle import TrussHandle

        truss_handle = args[0]

        if truss_handle.is_shadow:
            return func(*args, **kwargs)

        gathered_truss_handle = TrussHandle(
            truss_handle.gather(),
            original_truss_dir=truss_handle._truss_dir,
        )

        return func(gathered_truss_handle, *args[1:], **kwargs)

    return wrapper
