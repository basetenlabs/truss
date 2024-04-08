import builtins
import configparser
import contextlib
import enum
import logging
import socket
import sys
import textwrap
import time
import traceback
from typing import Any, Callable, Iterable, Iterator, NoReturn, Type, TypeVar

import fastapi
import httpx
import pydantic
from slay import definitions
from truss.remote import remote_factory

T = TypeVar("T")


@contextlib.contextmanager
def log_level(level: int) -> Iterator[None]:
    """Change loglevel for code in this context."""
    current_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(current_logging_level)


def expect_one(it: Iterable[T]) -> T:
    """Assert that an iterable has exactly on element and return it."""
    it = iter(it)
    try:
        element = next(it)
    except StopIteration:
        raise ValueError("Iterable is empty.")

    try:
        _ = next(it)
    except StopIteration:
        return element

    raise ValueError("Iterable has more than one element.")


class ConditionStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    NOT_DONE = enum.auto()


def wait_for_condition(
    condition: Callable[[], ConditionStatus],
    retries: int = 10,
    sleep_between_retries_secs: int = 1,
) -> bool:
    for _ in range(retries):
        cond_status = condition()
        if cond_status == ConditionStatus.SUCCESS:
            return True
        if cond_status == ConditionStatus.FAILURE:
            return False
        time.sleep(sleep_between_retries_secs)
    return False


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        s.listen(1)  # Not necessary but included for completeness.
        port = s.getsockname()[1]  # Retrieve the port number assigned.
        return port


def get_api_key_from_trussrc() -> str:
    try:
        return remote_factory.load_config().get("baseten", "api_key")
    except configparser.Error as e:
        raise definitions.MissingDependencyError(
            "You must have a `trussrc` file with a baseten API key."
        ) from e


def call_workflow_dbg(
    service: definitions.ServiceDescriptor,
    payload: Any,
    max_retries: int = 100,
    retry_wait_sec: int = 3,
) -> httpx.Response:
    """For debugging only: tries calling a workflow."""
    api_key = get_api_key_from_trussrc()
    session = httpx.Client(headers={"Authorization": f"Api-Key {api_key}"})
    for _ in range(max_retries):
        try:
            response = session.post(service.predict_url, json=payload)
            return response
        except Exception:
            time.sleep(retry_wait_sec)
    raise


# Error Propagation Utils. ##############################################################


def _handle_exception(
    exception: Exception, include_stack: bool, processor_name: str
) -> NoReturn:
    """Raises `fastapi.HTTPException` with `RemoteErrorDetail` as detail."""
    if hasattr(exception, "__module__"):
        exception_module_name = exception.__module__
    else:
        exception_module_name = None

    if include_stack:
        error_stack = traceback.extract_tb(exception.__traceback__)
        # Exclude the error handling functions from the stack trace.
        exclude_frames = {exception_to_http_error.__name__, handle_response.__name__}
        final_tb = [frame for frame in error_stack if frame.name not in exclude_frames]
        stack = list(
            [definitions.StackFrame.from_frame_summary(frame) for frame in final_tb]
        )
    else:
        stack = []

    error = definitions.RemoteErrorDetail(
        remote_name=processor_name,
        exception_class_name=exception.__class__.__name__,
        exception_module_name=exception_module_name,
        exception_message=str(exception),
        user_stack_trace=stack,
    )
    raise fastapi.HTTPException(status_code=500, detail=error.dict())


@contextlib.contextmanager
def exception_to_http_error(include_stack: bool, processor_name: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        _handle_exception(e, include_stack, processor_name)


def _resolve_exception_class(
    error: definitions.RemoteErrorDetail,
) -> Type[Exception]:
    """Tries to find the exception class in builtins or imported libs,
    falls back to `definitions.GenericRemoteError` if not found."""
    exception_cls = None
    if error.exception_module_name is None:
        exception_cls = getattr(builtins, error.exception_class_name, None)
    else:
        if mod := sys.modules.get(error.exception_module_name):
            exception_cls = getattr(mod, error.exception_class_name, None)

    if exception_cls is None:
        logging.warning(
            f"Could not resolve exception with name `{error.exception_class_name}` "
            f"and module `{error.exception_module_name}` - fall back to "
            f"`{definitions.GenericRemoteException.__name__}`."
        )
        exception_cls = definitions.GenericRemoteException

    return exception_cls


def handle_response(response: httpx.Response) -> Any:
    """For successful requests returns JSON, otherwise raises error.

    If the response error contains `RemoteErrorDetail`, it tries to re-raise
    the same exception that was raised remotely and falls back to
    `GenericRemoteException` if the exception class could not be resolved.

    Exception messages are chained to trace back to the root cause, i.e. the first
    processor that raised an exception. E.g. the message might look like this:

    ```
    RemoteProcessorError in "Workflow"
    Traceback (most recent call last):
      File "/app/model/processor.py", line 112, in predict
        result = await self._processor.run(
      File "/app/model/processor.py", line 79, in run
        value += self._text_to_num.run(part)
      File "/packages/remote_stubs.py", line 21, in run
        json_result = self._remote.predict_sync(json_args)
      File "/packages/slay/stub.py", line 37, in predict_sync
        return utils.handle_response(
    ValueError: (showing remote errors, root message at the bottom)
    --> Preceding Remote Cause:
        RemoteProcessorError in "TextToNum"
        Traceback (most recent call last):
          File "/app/model/processor.py", line 113, in predict
            result = self._processor.run(data=payload["data"])
          File "/app/model/processor.py", line 54, in run
            generated_text = self._replicator.run(data)
          File "/packages/remote_stubs.py", line 7, in run
            json_result = self._remote.predict_sync(json_args)
          File "/packages/slay/stub.py", line 37, in predict_sync
            return utils.handle_response(
        ValueError: (showing remote errors, root message at the bottom)
        --> Preceding Remote Cause:
            RemoteProcessorError in "TextReplicator"
            Traceback (most recent call last):
              File "/app/model/processor.py", line 112, in predict
                result = self._processor.run(data=payload["data"])
              File "/app/model/processor.py", line 36, in run
                raise ValueError(f"This input is too long: {len(data)}.")
            ValueError: This input is too long: 100.

    ```
    """
    if response.is_server_error:
        try:
            response_json = response.json()
        except Exception as e:
            raise ValueError("Could not get JSON from error response") from e

        try:
            error_json = response_json["error"]
        except KeyError as e:
            raise ValueError(
                "Could not get error field from JSON from error response"
            ) from e

        try:
            error = definitions.RemoteErrorDetail.parse_obj(error_json)
        except pydantic.ValidationError as e:
            raise ValueError(f"Could not parse error: {error_json}") from e

        exception_cls = _resolve_exception_class(error)
        msg = (
            f"(showing remote errors, root message at the bottom)\n"
            f"--> Preceding Remote Cause:\n"
            f"{textwrap.indent(error.format(), '    ')}"
        )
        raise exception_cls(msg)

    if response.is_client_error:
        raise fastapi.HTTPException(
            status_code=response.status_code, detail=response.content
        )
    return response.json()
