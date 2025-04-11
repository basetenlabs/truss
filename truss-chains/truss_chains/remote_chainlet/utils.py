import asyncio
import builtins
import contextlib
import contextvars
import json
import logging
import sys
import textwrap
import threading
import traceback
from collections.abc import AsyncIterator
from typing import Any, Iterator, Mapping, NoReturn, Optional, Type, TypeVar

import aiohttp
import fastapi
import httpx
import pydantic

from truss.templates.shared import dynamic_config_resolver
from truss_chains import private_types, public_types

T = TypeVar("T")


def populate_chainlet_service_predict_urls(
    chainlet_to_service: Mapping[str, private_types.ServiceDescriptor],
) -> Mapping[str, public_types.DeployedServiceDescriptor]:
    chainlet_to_deployed_service: dict[str, public_types.DeployedServiceDescriptor] = {}
    if not chainlet_to_service:
        return {}

    dynamic_chainlet_config_str = dynamic_config_resolver.get_dynamic_config_value_sync(
        private_types.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    if not dynamic_chainlet_config_str:
        raise public_types.MissingDependencyError(
            f"No '{private_types.DYNAMIC_CHAINLET_CONFIG_KEY}' "
            "found. Cannot override Chainlet configs."
        )

    dynamic_chainlet_config = json.loads(dynamic_chainlet_config_str)

    for chainlet_name, service_descriptor in chainlet_to_service.items():
        display_name = service_descriptor.display_name

        # NOTE: The Chainlet `display_name` in the Truss CLI
        # corresponds to Chainlet `name` in the backend. As
        # the dynamic Chainlet config is keyed on the backend
        # Chainlet name, we have to look up config values by
        # using the `display_name` in the service descriptor.
        if display_name not in dynamic_chainlet_config:
            raise public_types.MissingDependencyError(
                f"Chainlet '{display_name}' not found in "
                f"'{private_types.DYNAMIC_CHAINLET_CONFIG_KEY}'. "
                f"Dynamic Chainlet config keys: {list(dynamic_chainlet_config)}."
            )

        if internal_url := dynamic_chainlet_config[display_name].get("internal_url"):
            url = {"internal_url": internal_url}
        else:
            predict_url = dynamic_chainlet_config[display_name].get("predict_url")
            url = {"predict_url": predict_url}

        chainlet_to_deployed_service[chainlet_name] = (
            public_types.DeployedServiceDescriptor(
                display_name=display_name,
                name=service_descriptor.name,
                options=service_descriptor.options,
                **url,
            )
        )

    return chainlet_to_deployed_service


class AsyncSafeCounter:
    def __init__(self, initial: int = 0) -> None:
        self._counter = initial
        self._lock = asyncio.Lock()

    async def increment(self) -> int:
        async with self._lock:
            self._counter += 1
            return self._counter

    async def decrement(self) -> int:
        async with self._lock:
            self._counter -= 1
            return self._counter

    async def __aenter__(self) -> int:
        return await self.increment()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.decrement()


class ThreadSafeCounter:
    def __init__(self, initial: int = 0) -> None:
        self._counter = initial
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def decrement(self) -> int:
        with self._lock:
            self._counter -= 1
            return self._counter

    def __enter__(self) -> int:
        return self.increment()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.decrement()


_trace_parent_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_parent", default=None
)


@contextlib.contextmanager
def _trace_parent(headers: Mapping[str, str]) -> Iterator[None]:
    token = _trace_parent_context.set(
        headers.get(private_types.OTEL_TRACE_PARENT_HEADER_KEY, "")
    )
    try:
        yield
    finally:
        _trace_parent_context.reset(token)


@contextlib.contextmanager
def trace_parent_raw(trace_parent: str) -> Iterator[None]:
    token = _trace_parent_context.set(trace_parent)
    try:
        yield
    finally:
        _trace_parent_context.reset(token)


def get_trace_parent() -> Optional[str]:
    return _trace_parent_context.get()


def pydantic_set_field_dict(obj: pydantic.BaseModel) -> dict[str, pydantic.BaseModel]:
    """Like `BaseModel.model_dump(exclude_unset=True), but only top-level.

    This is used to get kwargs for invoking a function, while dropping fields for which
    there is no value explicitly set in the pydantic model. A field is considered unset
    if the key was not present in the incoming JSON request (from which the model was
    parsed/initialized) and the pydantic model has a default value, such as `None`.

    By dropping these unset fields, the default values from the function definition
    will be used instead. This behavior ensures correct handling of arguments where
    the function has a default, such as in the case of `run_remote`. If the model has
    an optional field defaulting to `None`, this approach differentiates between
    the user explicitly passing a value of `None` and the field being unset in the
    request.

    """
    return {name: getattr(obj, name) for name in obj.model_fields_set}


# Error Propagation Utils. #############################################################


def _handle_exception(exception: Exception) -> NoReturn:
    """Raises `HTTPException` with `RemoteErrorDetail`."""
    if hasattr(exception, "__module__"):
        exception_module_name = exception.__module__
    else:
        exception_module_name = None

    error_stack = traceback.extract_tb(exception.__traceback__)
    # Filter everything before (model.py) and after (stubs, error handling) so that only
    # user-defined code remains. See test_e2e.py::test_chain for expected results.
    model_predict_index = 0
    first_stub_index = len(error_stack)
    for i, frame in enumerate(error_stack):
        if frame.filename.endswith("model/model.py") and frame.name == "predict":
            model_predict_index = i + 1
        if frame.filename.endswith("remote_chainlet/stub.py") and frame.name.startswith(
            "predict"  # predict sycnc|async|stream.
        ):
            first_stub_index = i - 1
            break

    final_tb = error_stack[model_predict_index:first_stub_index]
    stack = [
        public_types.RemoteErrorDetail.StackFrame.from_frame_summary(frame)
        for frame in final_tb
    ]
    error = public_types.RemoteErrorDetail(
        exception_cls_name=exception.__class__.__name__,
        exception_module_name=exception_module_name,
        exception_message=str(exception),
        user_stack_trace=list(stack),
    )
    if isinstance(exception, fastapi.HTTPException):
        status_code = exception.status_code
    else:
        status_code = fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR

    raise fastapi.HTTPException(
        status_code=status_code, detail=error.model_dump()
    ) from exception


def _resolve_exception_class(error: public_types.RemoteErrorDetail) -> Type[Exception]:
    """Tries to find the exception class in builtins or imported libs,
    falls back to `public_types.GenericRemoteError` if not found."""
    exception_cls = None
    if error.exception_module_name is None:
        exception_cls = getattr(builtins, error.exception_cls_name, None)
    else:
        if mod := sys.modules.get(error.exception_module_name):
            exception_cls = getattr(mod, error.exception_cls_name, None)

    if exception_cls is None:
        logging.warning(
            f"Could not resolve exception with name `{error.exception_cls_name}` "
            f"and module `{error.exception_module_name}` - fall back to "
            f"`{public_types.GenericRemoteException.__name__}`."
        )
        exception_cls = public_types.GenericRemoteException

    if issubclass(exception_cls, pydantic.ValidationError):
        # Cannot re-raise naively.
        # https://github.com/pydantic/pydantic/issues/6734.
        exception_cls = public_types.GenericRemoteException

    return exception_cls


def _handle_response_error(response_json: dict, base_msg: str, http_status: int):
    try:
        error_json = response_json["error"]
    except KeyError as e:
        logging.error(f"response_json: {response_json}")
        raise ValueError(
            f"{base_msg}. Could not get `error` field from JSON response."
        ) from e

    try:
        error = public_types.RemoteErrorDetail.model_validate(error_json)
    except pydantic.ValidationError as e:
        if isinstance(error_json, str):
            msg = f"{base_msg}: '{error_json}'"
            raise public_types.GenericRemoteException(msg) from None
        raise ValueError(
            f"{base_msg}: Could not parse chainlet error. Error details are expected "
            "to be either a plain string (old truss models) or a serialized "
            f"`{public_types.RemoteErrorDetail.__name__}`, got:\n{repr(error_json)}"
        ) from e

    exception_cls = _resolve_exception_class(error)
    error_format = textwrap.indent(error.format(), "│   ")
    *lines, last_line = error_format.splitlines()
    last_line = f"╰{last_line[1:]}" if last_line.startswith("│") else last_line
    error_format = "\n".join(lines + [last_line])
    msg = (
        f"(showing chained remote errors, root error at the bottom)\n"
        f"├─ {base_msg}\n"
        f"{error_format}"
    )

    if issubclass(exception_cls, fastapi.HTTPException):
        raise fastapi.HTTPException(status_code=http_status, detail=msg)

    try:
        exc = exception_cls(msg)
    except Exception:  # ruff
        # Some exceptions cannot be directly instantiated with message, fallback.
        exc = public_types.GenericRemoteException(msg)

    raise exc


def _make_base_error_message(remote_name: str, http_status: int) -> str:
    return (
        f"Error calling dependency Chainlet `{remote_name}`, "
        f"HTTP status={http_status}, trace ID=`{get_trace_parent()}`."
    )


def response_raise_errors(response: httpx.Response, remote_name: str) -> None:
    """In case of error, raise it.

    If the response error contains `RemoteErrorDetail`, it tries to re-raise
    the same exception that was raised remotely and falls back to
    `GenericRemoteException` if the exception class could not be resolved.

    Exception messages are chained to trace back to the root cause, i.e. the first
    Chainlet that raised an exception. E.g. the message might look like this:

    ```
    Chainlet-Traceback (most recent call last):
      File "/packages/itest_chain.py", line 132, in run_remote
        value = self._accumulate_parts(text_parts.parts)
      File "/packages/itest_chain.py", line 144, in _accumulate_parts
        value += self._text_to_num.run_remote(part)
    ValueError: (showing chained remote errors, root error at the bottom)
    ├─ Error in dependency Chainlet `TextToNum` (HTTP status 500):
    │   Chainlet-Traceback (most recent call last):
    │     File "/packages/itest_chain.py", line 87, in run_remote
    │       generated_text = self._replicator.run_remote(data)
    │   ValueError: (showing chained remote errors, root error at the bottom)
    │   ├─ Error in dependency Chainlet `TextReplicator` (HTTP status 500):
    │   │   Chainlet-Traceback (most recent call last):
    │   │     File "/packages/itest_chain.py", line 52, in run_remote
    │   │       validate_data(data)
    │   │     File "/packages/itest_chain.py", line 36, in validate_data
    │   │       raise ValueError(f"This input is too long: {len(data)}.")
    ╰   ╰   ValueError: This input is too long: 100.
    ```
    """
    if response.is_error:
        base_msg = _make_base_error_message(remote_name, response.status_code)
        try:
            response_json = response.json()
        except Exception as e:
            raise ValueError(base_msg) from e
        _handle_response_error(response_json, base_msg, response.status_code)


async def async_response_raise_errors(
    response: aiohttp.ClientResponse, remote_name: str
) -> None:
    """Async version of `async_response_raise_errors`."""
    if response.status >= 400:
        base_msg = _make_base_error_message(remote_name, response.status)
        try:
            response_json = await response.json()
        except Exception as e:
            raise ValueError(base_msg) from e
        _handle_response_error(response_json, base_msg, response.status)


@contextlib.contextmanager
def predict_context(headers: Mapping[str, str]) -> Iterator[None]:
    with _trace_parent(headers):
        try:
            yield
        except Exception as e:
            _handle_exception(e)


class WebsocketWrapperFastAPI:
    """Implements `private_types.WebSocketProtocol` around fastAPI object."""

    # TODO: consider if we want to wrap/translate exceptions thrown as well. Currently
    #  this is somewhat loopy, as `fastapi.WebSocketDisconnect` just passes through,
    #  but we have not documented either, so it's not directly a contradiction.

    def __init__(self, websocket: fastapi.WebSocket) -> None:
        self._websocket = websocket
        self.headers: Mapping[str, str] = websocket.headers

    async def close(self, code: int = 1000, reason: Optional[str] = None) -> None:
        await self._websocket.close(code=code, reason=reason)

    async def receive_text(self) -> str:
        return await self._websocket.receive_text()

    async def receive_bytes(self) -> bytes:
        return await self._websocket.receive_bytes()

    async def receive_json(self) -> Any:
        return await self._websocket.receive_json()

    async def send_text(self, data: str) -> None:
        await self._websocket.send_text(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._websocket.send_bytes(data)

    async def send_json(self, data: Any) -> None:
        await self._websocket.send_json(data)

    async def iter_text(self) -> AsyncIterator[str]:
        while True:
            yield await self.receive_text()

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        while True:
            yield await self.receive_bytes()

    async def iter_json(self) -> AsyncIterator[Any]:
        while True:
            yield await self.receive_json()
