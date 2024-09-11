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


# See https://github.com/basetenlabs/baseten/blob/master/docs/Error-Propagation.md
_BASETEN_UNEXPECTED_ERROR = 500
_BASETEN_DOWNSTREAM_ERROR_CODE = 600
_BASETEN_CLIENT_ERROR_CODE = 700
_USER_RAISED_HTTP_EXCEPTION_ATTR = "_USER_RAISED_HTTP_EXCEPTION_ATTR"


def _make_baseten_response(
    http_status: int,
    info: Union[str, Exception],
    baseten_error_code: int,
) -> fastapi.Response:
    msg = str(info) if isinstance(info, Exception) else info
    return JSONResponse(
        status_code=http_status,
        content={"error": msg},
        headers=_make_baseten_error_headers(baseten_error_code),
    )


async def exception_handler(
    request: fastapi.Request, exc: Exception
) -> fastapi.Response:
    if isinstance(exc, ModelMissingError):
        return _make_baseten_response(
            HTTPStatus.NOT_FOUND.value, exc, _BASETEN_DOWNSTREAM_ERROR_CODE
        )
    if isinstance(exc, ModelNotReady):
        return _make_baseten_response(
            HTTPStatus.SERVICE_UNAVAILABLE.value, exc, _BASETEN_DOWNSTREAM_ERROR_CODE
        )
    if isinstance(exc, InputParsingError):
        return _make_baseten_response(
            HTTPStatus.BAD_REQUEST.value,
            exc,
            _BASETEN_CLIENT_ERROR_CODE,
        )
    if isinstance(exc, UserCodeError):
        return _make_baseten_response(
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "Internal Server Error",
            _BASETEN_DOWNSTREAM_ERROR_CODE,
        )
    if isinstance(exc, fastapi.HTTPException):
        # This is a pass through, but additionally adds our custom error headers.
        return _make_baseten_response(
            exc.status_code, exc.detail, _BASETEN_DOWNSTREAM_ERROR_CODE
        )

    # This case should never happen
    return _make_baseten_response(
        HTTPStatus.INTERNAL_SERVER_ERROR.value,
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        _BASETEN_UNEXPECTED_ERROR,
    )


HANDLED_EXCEPTIONS = {
    ModelMissingError,
    ModelNotReady,
    NotImplementedError,
    InputParsingError,
    UserCodeError,
    fastapi.HTTPException,
}


def _intercept_user_exception(exc: Exception, logger: logging.Logger) -> NoReturn:
    # Note that logger.exception logs the stacktrace, such that the user can
    # debug this error from the logs.
    # TODO: consider removing the wrapper function from the stack trace.
    if isinstance(exc, HTTPException):
        logger.exception("Model raised HTTPException", stacklevel=2)
        setattr(exc, _USER_RAISED_HTTP_EXCEPTION_ATTR, True)
        raise exc
    else:
        logger.exception("Internal Server Error", stacklevel=2)
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
                _intercept_user_exception(e, logger)

        return inner_async  # type: ignore[return-value]
    else:
        # Handle synchronous function
        def inner_sync(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _intercept_user_exception(e, logger)

        return inner_sync
