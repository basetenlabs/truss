def proxy_to_shadow_if_scattered(func):
    def wrapper(*args, **kwargs):
        from truss.truss_handle import TrussHandle

        truss_handle = args[0]
        if not truss_handle.is_scattered():
            return func(*args, **kwargs)

        gathered_truss_handle = TrussHandle(truss_handle.gather())
        return func(gathered_truss_handle, *args[1:], **kwargs)

    return wrapper


def in_progress_state_indicator(func):
    def wrapper(*args, **kwargs):
        from halo import Halo

        with Halo(
            text=f"Preparing {str(func.__name__).split('_')[0]} context",
            spinner="star",
            color="green",
        ):
            return func(*args, **kwargs)

    return wrapper
