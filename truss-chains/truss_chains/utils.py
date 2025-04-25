import contextlib
import enum
import inspect
import logging
import os
import random
from typing import Any, Iterable, Iterator, TypeVar, Union

import pydantic
import pydantic_core

from truss_chains import public_types

T = TypeVar("T")


def make_abs_path_here(file_path: str) -> "public_types.AbsPath":
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

    return public_types.AbsPath(abs_file_path, module_path, file_path)


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


def get_pydantic_field_default_value(
    model: type[pydantic.BaseModel], field_name: str
) -> Any:
    """Retrieve the default value of a field, considering both default and default_factory."""
    field_info = model.model_fields[field_name]
    if field_info.default is not pydantic_core.PydanticUndefined:
        return field_info.default
    if field_info.default_factory is not None:
        return field_info.default_factory()  #  type: ignore[call-arg]
    return None


def make_optional_import_error(module_name: str) -> public_types.ChainsRuntimeError:
    return public_types.ChainsRuntimeError(
        f"Could not import `{module_name}`. For chains CLI (truss package) this is an "
        "optional dependency. In deployed chainlets this dependency is "
        "automatically added. If you happen to run into this error, "
        f"install `{module_name}` manually."
    )
