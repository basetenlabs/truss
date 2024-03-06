import contextlib
import io
import json
import logging
from typing import Iterable, TypeVar

import pydantic

T = TypeVar("T")


@contextlib.contextmanager
def log_level(level: int):
    current_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(current_logging_level)


@contextlib.contextmanager
def no_print():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def pydantic_pp(model: pydantic.BaseModel) -> str:
    return json.dumps(model.model_dump(), indent=4)


def expect_one(it: Iterable[T]) -> T:
    it = iter(it)
    try:
        element = next(it)
    except StopIteration:
        raise ValueError("Empty")

    try:
        other = next(it)
    except StopIteration:
        return element

    raise ValueError("Contains other.")
