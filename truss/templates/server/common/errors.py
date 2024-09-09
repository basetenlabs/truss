import asyncio
import logging
from http import HTTPStatus
from typing import (
    Callable,
    Coroutine,
    Mapping,
    NoReturn,
    Optional,
    TypeVar,
    Union,
    overload,
)

import fastapi
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing_extensions import ParamSpec


class ModelMissingError(Exception):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return self.path


class ModelNotReady(RuntimeError):
    def __init__(self, model_name: str, detail: Optional[str] = None):
        self.model_name = model_name
        self.error_msg = f"Model with name {self.model_name} is not ready."
        if detail:
            self.error_msg = self.error_msg + " " + detail

    def __str__(self):
        return self.error_msg


class InputParsingError(ValueError):
    pass


class UserCodeError(Exception):
    pass


def _make_baseten_error_headers(error_code: int) -> Mapping[str, str]:
    # The source of truth for these constants is in
    # go/beefeater/shared/error_propagated.go
    truss_server_service_id = "04"
    return {
        "X-BASETEN-ERROR-SOURCE": truss_server_service_id,
        "X-BASETEN-ERROR-CODE": str(error_code),
    }


def _make_baseten_response(
    http_status: int,
    info: Union[str, Exception],
    baseten_error_code: Optional[int] = None,
) -> fastapi.Response:
    msg = str(info) if isinstance(info, Exception) else info

    error_code = baseten_error_code if baseten_error_code is not None else http_status

    return JSONResponse(
        status_code=http_status,
        content={"error": msg},
        headers=_make_baseten_error_headers(error_code),
    )


async def generic_exception_handler(_, exc):
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content={"error": f"{type(exc).__name__} : {str(exc)}"},
    )


async def exception_handler(
    request: fastapi.Request, exc: Exception
) -> fastapi.Response:
    if isinstance(exc, ModelMissingError):
        return _make_baseten_response(HTTPStatus.NOT_FOUND, exc)
    if isinstance(exc, ModelNotReady):
        return _make_baseten_response(HTTPStatus.SERVICE_UNAVAILABLE, exc)
    if isinstance(exc, NotImplementedError):
        return _make_baseten_response(HTTPStatus.NOT_IMPLEMENTED, exc)
    if isinstance(exc, InputParsingError):
        return _make_baseten_response(HTTPStatus.BAD_REQUEST, exc)
    if isinstance(exc, UserCodeError):
        # TODO: need a specific code?
        return _make_baseten_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, "Internal Server Error"
        )
    if isinstance(exc, fastapi.HTTPException):
        # This is a pass through, but additionally adds our custom error headers.
        return _make_baseten_response(exc.status_code, exc.detail)

    # Any other exceptions will be turned into "internal server error".
    return _make_baseten_response(HTTPStatus.INTERNAL_SERVER_ERROR, exc)


def _handle_exception(exc: Exception, logger: logging.Logger) -> NoReturn:
    # Note that logger.exception logs the stacktrace, such that the user can
    # debug this error from the logs. We use this decorator to ensure that we
    # only log the stack trace from here and not include all the server code above.
    if isinstance(exc, HTTPException):
        logger.exception("Model raised HTTPException")
        raise exc
    else:
        logger.exception("Internal Server Error")
        raise UserCodeError(str(exc))


_P = ParamSpec("_P")
_R = TypeVar("_R")
_R_async = TypeVar("_R_async", bound=Coroutine)  # Return type for async functions


@overload
def intercept_exceptions(
    func: Callable[_P, _R], logger: logging.Logger
) -> Callable[_P, _R]: ...


@overload
def intercept_exceptions(
    func: Callable[_P, _R_async], logger: logging.Logger
) -> Callable[_P, _R_async]: ...


def intercept_exceptions(
    func: Callable[_P, _R], logger: logging.Logger
) -> Callable[_P, _R]:
    """Converts all exceptions to 500-`HTTPException` and logs them.
    If exception is already `HTTPException`, re-raises exception as is.
    """
    if asyncio.iscoroutinefunction(func):
        # Handle asynchronous function
        async def inner_async(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                _handle_exception(e, logger)

        return inner_async  # type: ignore[return-value]
    else:
        # Handle synchronous function
        def inner_sync(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _handle_exception(e, logger)

        return inner_sync
