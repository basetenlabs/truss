import asyncio
import builtins
import contextlib
import enum
import inspect
import json
import logging
import os
import random
import socket
import sys
import textwrap
import threading
import traceback
from typing import Any, Iterable, Iterator, Mapping, NoReturn, Type, TypeVar, Union

import fastapi
import httpx
import pydantic
from truss.templates.shared import dynamic_config_resolver

from truss_chains import definitions

T = TypeVar("T")


def make_abs_path_here(file_path: str) -> definitions.AbsPath:
    """Helper to specify file paths relative to the *immediately calling* module.

    E.g. in you have a project structure like this::

        root/
            chain.py
            common_requirements.text
            sub_package/
                chainlet.py
                chainlet_requirements.txt

    You can now in ``root/sub_package/chainlet.py`` point to the requirements
    file like this::

        shared = make_abs_path_here("../common_requirements.text")
        specific = make_abs_path_here("chainlet_requirements.text")


    Warning:
        This helper uses the directory of the immediately calling module as an
        absolute reference point for resolving the file location. Therefore,
        you MUST NOT wrap the instantiation of ``make_abs_path_here`` into a
        function (e.g. applying decorators) or use dynamic code execution.

        Ok::

            def foo(path: AbsPath):
                abs_path = path.abs_path


            foo(make_abs_path_here("./somewhere"))

        Not Ok::

            def foo(path: str):
                dangerous_value = make_abs_path_here(path).abs_path


            foo("./somewhere")

    """
    # TODO: the absolute path resolution below uses the calling module as a
    #   reference point. This would not work if users wrap this call in a function
    #   - we hope the naming makes clear that this should not be done.
    caller_frame = inspect.stack()[1]
    module_path = caller_frame.filename
    if not os.path.isabs(file_path):
        module_dir = os.path.dirname(os.path.abspath(module_path))
        abs_file_path = os.path.normpath(os.path.join(module_dir, file_path))
    else:
        abs_file_path = file_path

    return definitions.AbsPath(abs_file_path, module_path, file_path)


def setup_dev_logging(level: Union[int, str] = logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    log_format = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    date_format = "%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


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


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        s.listen(1)  # Not necessary but included for completeness.
        port = s.getsockname()[1]  # Retrieve the port number assigned.
        return port


def override_chainlet_to_service_metadata(
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
) -> Mapping[str, definitions.ServiceDescriptor]:
    # Override predict_urls in chainlet_to_service ServiceDescriptors if chainlet_url_map exists in dynamic config
    chainlet_url_map_str = dynamic_config_resolver.get_dynamic_config_value(
        definitions.DYNAMIC_CONFIG_CHAINLET_URL_MAP_KEY
    )
    if chainlet_url_map_str:
        chainlet_url_map = json.loads(chainlet_url_map_str)
        for (
            chainlet_name,
            service_descriptor,
        ) in chainlet_to_service.items():
            if chainlet_name in chainlet_url_map:
                # We update the predict_url to be the one pulled from the dynamic config url mapping
                service_descriptor.predict_url = chainlet_url_map[chainlet_name]
    return chainlet_to_service


# Error Propagation Utils. #############################################################


def _handle_exception(
    exception: Exception, include_stack: bool, chainlet_name: str
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
        remote_name=chainlet_name,
        exception_cls_name=exception.__class__.__name__,
        exception_module_name=exception_module_name,
        exception_message=str(exception),
        user_stack_trace=stack,
    )
    raise fastapi.HTTPException(
        status_code=500, detail=error.model_dump()
    ) from exception


@contextlib.contextmanager
def exception_to_http_error(include_stack: bool, chainlet_name: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        _handle_exception(e, include_stack, chainlet_name)


def _resolve_exception_class(
    error: definitions.RemoteErrorDetail,
) -> Type[Exception]:
    """Tries to find the exception class in builtins or imported libs,
    falls back to `definitions.GenericRemoteError` if not found."""
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
            f"`{definitions.GenericRemoteException.__name__}`."
        )
        exception_cls = definitions.GenericRemoteException

    return exception_cls


def handle_response(response: httpx.Response, remote_name: str) -> Any:
    """For successful requests returns JSON, otherwise raises error.

    If the response error contains `RemoteErrorDetail`, it tries to re-raise
    the same exception that was raised remotely and falls back to
    `GenericRemoteException` if the exception class could not be resolved.

    Exception messages are chained to trace back to the root cause, i.e. the first
    Chainlet that raised an exception. E.g. the message might look like this:

    ```
    RemoteChainletError in "Chain"
    Traceback (most recent call last):
      File "/app/model/Chainlet.py", line 112, in predict
        result = await self._chainlet.run(
      File "/app/model/Chainlet.py", line 79, in run
        value += self._text_to_num.run(part)
      File "/packages/remote_stubs.py", line 21, in run
        json_result = self._remote.predict_sync(json_args)
      File "/packages/truss_chains/stub.py", line 37, in predict_sync
        return utils.handle_response(
    ValueError: (showing remote errors, root message at the bottom)
    --> Preceding Remote Cause:
        RemoteChainletError in "TextToNum"
        Traceback (most recent call last):
          File "/app/model/Chainlet.py", line 113, in predict
            result = self._chainlet.run(data=payload["data"])
          File "/app/model/Chainlet.py", line 54, in run
            generated_text = self._replicator.run(data)
          File "/packages/remote_stubs.py", line 7, in run
            json_result = self._remote.predict_sync(json_args)
          File "/packages/truss_chains/stub.py", line 37, in predict_sync
            return utils.handle_response(
        ValueError: (showing remote errors, root message at the bottom)
        --> Preceding Remote Cause:
            RemoteChainletError in "TextReplicator"
            Traceback (most recent call last):
              File "/app/model/Chainlet.py", line 112, in predict
                result = self._chainlet.run(data=payload["data"])
              File "/app/model/Chainlet.py", line 36, in run
                raise ValueError(f"This input is too long: {len(data)}.")
            ValueError: This input is too long: 100.

    ```
    """
    if response.is_error:
        try:
            response_json = response.json()
        except Exception as e:
            raise ValueError(
                "Could not get JSON from error response. Status: "
                f"`{response.status_code}`."
            ) from e

        try:
            error_json = response_json["error"]
        except KeyError as e:
            logging.error(f"response.json(): {response_json}")
            raise ValueError(
                "Could not get `error` field from JSON from error response"
            ) from e

        try:
            error = definitions.RemoteErrorDetail.model_validate(error_json)
        except pydantic.ValidationError as e:
            if isinstance(error_json, str):
                msg = f"Remote error occurred in `{remote_name}`: '{error_json}'"
                raise definitions.GenericRemoteException(msg) from None
            raise ValueError(
                "Could not parse error. Error details are expected to be either a "
                "plain string (old truss models) or a serialized "
                f"`definitions.RemoteErrorDetail.__name__`, got:\n{repr(error_json)}"
            ) from e

        exception_cls = _resolve_exception_class(error)
        msg = (
            f"(showing remote errors, root message at the bottom)\n"
            f"--> Preceding Remote Cause:\n"
            f"{textwrap.indent(error.format(), '    ')}"
        )
        raise exception_cls(msg)

    return response.json()


class InjectedError(Exception):
    """Test error for debugging/dev."""


def random_fail(probability: float, msg: str):
    """Probabilistically raises `InjectedError` for debugging/dev."""
    if random.random() < probability:
        print(f"Random failure: {msg}")
        raise InjectedError(msg)


class StrEnum(str, enum.Enum):
    """
    Adapted from MIT-licensed
    https://github.com/irgeek/StrEnum/blob/master/strenum/__init__.py

    This is useful for Pydantic-based (de-)serialisation, as Pydantic takes the value
    of an enum member as the value to be (de-)serialised, and not the name of the
    member. With this, we can have the member name and value be the same by using
    `enum.auto()`.

    StrEnum is a Python `enum.Enum` that inherits from `str`. The `auto()` behavior
    uses the member name and lowers it. This is useful for compatibility with pydantic.
    Example usage:

    ```
    class Example(StrEnum):
        SOME_VALUE = enum.auto()
        ANOTHER_VALUE = enum.auto()
        TEST = enum.auto()

    assert Example.SOME_VALUE == "SOME_VALUE"
    assert Example.ANOTHER_VALUE.value == "ANOTHER_VALUE"
    assert Example.TEST.value == Example.TEST
    assert Example.TEST == Example("TEST")
    ```
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, str):
            raise TypeError(f"Values of StrEnums must be strings: Got `{repr(value)}`.")
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self) -> str:
        return str(self.value)

    def _generate_next_value_(name, *_) -> str:  # type: ignore[override]
        if name.upper() != name:
            raise ValueError(f"Python enum members should be upper case. Got `{name}`.")
        return name


def issubclass_safe(x: Any, cls: type) -> bool:
    """Like built-in `issubclass`, but works on non-type objects."""
    return isinstance(x, type) and issubclass(x, cls)


def pydantic_set_field_dict(obj: pydantic.BaseModel) -> dict[str, pydantic.BaseModel]:
    """Like `BaseModel.model_dump(exclude_unset=True), but only top-level."""
    return {name: getattr(obj, name) for name in obj.__fields_set__}


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
