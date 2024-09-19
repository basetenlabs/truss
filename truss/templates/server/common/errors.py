import contextlib
import logging
import sys
from http import HTTPStatus
from types import TracebackType
from typing import (
    Generator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import fastapi
import starlette.responses
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# See https://github.com/basetenlabs/baseten/blob/master/docs/Error-Propagation.md
_TRUSS_SERVER_SERVICE_ID = 4
_BASETEN_UNEXPECTED_ERROR = 500
_BASETEN_DOWNSTREAM_ERROR_CODE = 600
_BASETEN_CLIENT_ERROR_CODE = 700


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


class ModelDefinitionError(TypeError):
    """When the user-defined truss model does not meet the contract."""


def _make_baseten_error_headers(error_code: int) -> Mapping[str, str]:
    return {
        "X-BASETEN-ERROR-SOURCE": f"{_TRUSS_SERVER_SERVICE_ID:02}",
        "X-BASETEN-ERROR-CODE": f"{error_code:03}",
    }


def add_error_headers_to_user_response(response: starlette.responses.Response) -> None:
    response.headers.update(_make_baseten_error_headers(_BASETEN_CLIENT_ERROR_CODE))


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


async def exception_handler(_: fastapi.Request, exc: Exception) -> fastapi.Response:
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
    if isinstance(exc, ModelDefinitionError):
        return _make_baseten_response(
            HTTPStatus.PRECONDITION_FAILED.value,
            f"{type(exc).__name__}: {str(exc)}",
            _BASETEN_DOWNSTREAM_ERROR_CODE,
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
    # Fallback case.
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
    ModelDefinitionError,
    fastapi.HTTPException,
}


def filter_traceback(
    model_file_name: str,
) -> Union[
    Tuple[Type[BaseException], BaseException, TracebackType],
    Tuple[None, None, None],
]:
    exc_type, exc_value, tb = sys.exc_info()
    if tb is None:
        return exc_type, exc_value, tb  # type: ignore[return-value]

    # Walk the traceback until we find the frame ending with 'model.py'
    current_tb: Optional[TracebackType] = tb
    while current_tb is not None:
        filename = current_tb.tb_frame.f_code.co_filename
        if filename.endswith(model_file_name):
            # Return exception info with traceback starting from current_tb
            return exc_type, exc_value, current_tb  # type: ignore[return-value]
        current_tb = current_tb.tb_next

    # If `model_file_name` not found, return the original exception info
    return exc_type, exc_value, tb  # type: ignore[return-value]


@contextlib.contextmanager
def intercept_exceptions(
    logger: logging.Logger, model_file_name: str
) -> Generator[None, None, None]:
    try:
        yield
    # Note that logger.error logs the stacktrace, such that the user can
    # debug this error from the logs.
    except HTTPException:
        logger.error(
            "Model raised HTTPException", exc_info=filter_traceback(model_file_name)
        )
        raise
    except Exception as exc:
        logger.error(
            "Internal Server Error", exc_info=filter_traceback(model_file_name)
        )
        raise UserCodeError(str(exc))
