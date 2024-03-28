import contextlib
import enum
import logging
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


@contextlib.contextmanager
def log_level(level: int):
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
