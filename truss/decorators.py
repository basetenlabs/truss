def proxy_to_shadow(func):
    def wrapper(*args, **kwargs):
        truss_handle = args[0]
        if not truss_handle.is_scattered():
            return func(*args, **kwargs)

        gathered_truss_handle = truss_handle.gather()
        return func(gathered_truss_handle, *args[1:], **kwargs)

    return wrapper
