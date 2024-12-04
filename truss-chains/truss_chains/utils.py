import asyncio
import contextlib
import enum
import inspect
import json
import logging
import os
import random
import socket
import threading
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    Union,
)

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


def populate_chainlet_service_predict_urls(
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
) -> Mapping[str, definitions.DeployedServiceDescriptor]:
    chainlet_to_deployed_service: Dict[str, definitions.DeployedServiceDescriptor] = {}

    dynamic_chainlet_config_str = dynamic_config_resolver.get_dynamic_config_value_sync(
        definitions.DYNAMIC_CHAINLET_CONFIG_KEY
    )

    if not dynamic_chainlet_config_str:
        raise definitions.MissingDependencyError(
            f"No '{definitions.DYNAMIC_CHAINLET_CONFIG_KEY}' found. Cannot override Chainlet configs."
        )

    dynamic_chainlet_config = json.loads(dynamic_chainlet_config_str)

    for (
        chainlet_name,
        service_descriptor,
    ) in chainlet_to_service.items():
        display_name = service_descriptor.display_name

        # NOTE: The Chainlet `display_name` in the Truss CLI
        # corresponds to Chainlet `name` in the backend. As
        # the dynamic Chainlet config is keyed on the backend
        # Chainlet name, we have to look up config values by
        # using the `display_name` in the service descriptor.
        if display_name not in dynamic_chainlet_config:
            raise definitions.MissingDependencyError(
                f"Chainlet '{display_name}' not found in '{definitions.DYNAMIC_CHAINLET_CONFIG_KEY}'. Dynamic Chainlet config keys: {list(dynamic_chainlet_config)}."
            )

        chainlet_to_deployed_service[chainlet_name] = (
            definitions.DeployedServiceDescriptor(
                display_name=display_name,
                name=service_descriptor.name,
                options=service_descriptor.options,
                predict_url=dynamic_chainlet_config[display_name]["predict_url"],
            )
        )

    return chainlet_to_deployed_service


########################################################################################


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
