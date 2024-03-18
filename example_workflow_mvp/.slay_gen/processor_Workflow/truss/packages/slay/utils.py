import contextlib
import io
import json
import logging
from typing import Callable, Iterable, TypeVar

import pydantic
from slay.deploy_truss import _ConditionStatus

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


def wait_for_condition(
    condition: Callable[[], _ConditionStatus],
    retries: int = 10,
    sleep_between_retries_secs: int = 1,
) -> bool:
    for _ in range(retries):
        cond_status = condition()
        if cond_status == _ConditionStatus.SUCCESS:
            return True
        if cond_status == _ConditionStatus.FAILURE:
            return False
        time.sleep(sleep_between_retries_secs)
    return False
