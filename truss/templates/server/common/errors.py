import contextlib
import logging
import sys
import textwrap
from http import HTTPStatus
from types import TracebackType
from typing import Generator, Mapping, Optional, Tuple, Type, Union

import fastapi
import pydantic
import starlette.responses
from fastapi.responses import JSONResponse

# See https://github.com/basetenlabs/baseten/blob/master/docs/Error-Propagation.md
_TRUSS_SERVER_SERVICE_ID = 4
_BASETEN_UNEXPECTED_ERROR = 500
_BASETEN_DOWNSTREAM_ERROR_CODE = 600
_BASETEN_CLIENT_ERROR_CODE = 700

MODEL_ERROR_MESSAGE = "Internal Server Error (in model/chainlet)."


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


class ModelMethodNotImplemented(Exception):
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
    http_status: int, info: Union[str, Exception], baseten_error_code: int
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
            HTTPStatus.BAD_REQUEST.value, exc, _BASETEN_CLIENT_ERROR_CODE
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
            MODEL_ERROR_MESSAGE,
            _BASETEN_DOWNSTREAM_ERROR_CODE,
        )
    if isinstance(exc, ModelMethodNotImplemented):
        return _make_baseten_response(
            HTTPStatus.NOT_FOUND.value, exc, _BASETEN_CLIENT_ERROR_CODE
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
    ModelMethodNotImplemented,
}


def filter_traceback(
    model_file_name: str,
) -> Union[
    Tuple[Type[BaseException], BaseException, TracebackType], Tuple[None, None, None]
]:
    exc_type, exc_value, tb = sys.exc_info()
    if tb is None:
        return exc_type, exc_value, tb  # type: ignore[return-value]

    # Store the last occurrence of the traceback matching the condition
    last_matching_tb: Optional[TracebackType] = None
    current_tb: Optional[TracebackType] = tb

    while current_tb is not None:
        filename = current_tb.tb_frame.f_code.co_filename
        if filename.endswith(model_file_name):
            last_matching_tb = current_tb
        current_tb = current_tb.tb_next

    # If a match was found, truncate the traceback at the last occurrence
    if last_matching_tb is not None:
        return exc_type, exc_value, last_matching_tb  # type: ignore[return-value]

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
    except fastapi.HTTPException as e:
        # TODO: we try to avoid any dependency of the truss server on chains, but for
        #  the purpose of getting readable chained-stack traces in the server logs,
        #  we have to add a special-case here.
        if "user_stack_trace" in e.detail:
            try:
                try:
                    from truss_chains import public_types

                    error_cls = public_types.RemoteErrorDetail
                except ImportError:
                    # For pre 0.9.67 usage.
                    from truss_chains import definitions  # type: ignore[attr-defined]

                    error_cls = definitions.RemoteErrorDetail

                chains_error = error_cls.model_validate(e.detail)
                # The formatted error contains a (potentially chained) stack trace
                # with all framework code removed, see
                # truss_chains/remote_chainlet/utils.py::response_raise_errors.
                logger.error(f"Chainlet raised Exception:\n{chains_error.format()}")
            except:  # If we cannot import chains or parse the error.
                logger.error(
                    "Model raised HTTPException",
                    exc_info=filter_traceback(model_file_name),
                )
                raise
            # If error was extracted successfully, the customized stack trace is
            # already printed above, so we raise with a clear traceback.
            e.__traceback__ = None
            raise e from None

        logger.error(
            "Model raised HTTPException", exc_info=filter_traceback(model_file_name)
        )
        raise
    except Exception as exc:
        logger.error(MODEL_ERROR_MESSAGE, exc_info=filter_traceback(model_file_name))
        raise UserCodeError(str(exc))


def _loc_to_dot_sep(loc: Tuple[Union[str, int], ...]) -> str:
    # From https://docs.pydantic.dev/latest/errors/errors/#customize-error-messages.
    # Chained field access is stylized with `.`-notation (corresponding to str parts)
    # and array indexing is stylized with `[i]`-notation (corresponding to int parts).
    # E.g. ('items', 1, 'value') -> items[1].value.
    parts = []
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                parts.append(".")
            parts.append(x)
        elif isinstance(x, int):
            parts.append(f"[{x}]")
        else:
            raise TypeError(f"Unexpected type: {x}.")
    return "".join(parts)


def format_pydantic_validation_error(exc: pydantic.ValidationError) -> str:
    if exc.error_count() == 1:
        parts = ["Input Parsing Error:"]
    else:
        parts = ["Input Parsing Errors:"]

    try:
        for error in exc.errors(include_url=False, include_input=False):
            loc = _loc_to_dot_sep(error["loc"])
            if error["msg"]:
                msg = error["msg"]
            else:
                error_no_loc = dict(error)
                error_no_loc.pop("loc")
                msg = str(error_no_loc)
            parts.append(textwrap.indent(f"`{loc}`: {msg}.", "  "))
    except:  # noqa: E722
        # Fallback in case any of the fields cannot be processed as expected.
        return f"Input Parsing Error, {str(exc)}"

    return "\n".join(parts)
