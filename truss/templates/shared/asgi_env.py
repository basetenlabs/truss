import os

TRUSS_ASGI_SERVER_ENV = "TRUSS_ASGI_SERVER"
TRUSS_ASYNC_BACKEND_ENV = "TRUSS_ASYNC_BACKEND"


def use_hypercorn_trio() -> bool:
    return (
        os.environ.get(TRUSS_ASGI_SERVER_ENV, "uvicorn").lower() == "hypercorn"
        and os.environ.get(TRUSS_ASYNC_BACKEND_ENV, "asyncio").lower() == "trio"
    )


def validate_asgi_backend_env() -> None:
    asgi = os.environ.get(TRUSS_ASGI_SERVER_ENV, "uvicorn").lower()
    backend = os.environ.get(TRUSS_ASYNC_BACKEND_ENV, "asyncio").lower()
    if backend == "trio" and asgi != "hypercorn":
        raise ValueError(
            f"{TRUSS_ASYNC_BACKEND_ENV}=trio requires {TRUSS_ASGI_SERVER_ENV}=hypercorn"
        )
    if asgi == "hypercorn" and backend != "trio":
        raise ValueError(
            f"{TRUSS_ASGI_SERVER_ENV}=hypercorn requires {TRUSS_ASYNC_BACKEND_ENV}=trio"
        )
