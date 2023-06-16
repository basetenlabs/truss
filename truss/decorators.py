def proxy_to_shadow(func):
    def wrapper(*args, **kwargs):
        from truss.truss_handle import TrussHandle
        from truss.util.path import are_dirs_equal

        truss_handle = args[0]
        gathered_truss_handle = TrussHandle(truss_handle.gather(), is_shadow_truss=True)

        if are_dirs_equal(truss_handle._truss_dir, gathered_truss_handle._truss_dir):
            return func(gathered_truss_handle, *args[1:], **kwargs)

        return func(*args, **kwargs)

    return wrapper
