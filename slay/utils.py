import configparser
import contextlib
import enum
import logging
import socket
import time
from typing import Any, Callable, Iterable, TypeVar

import httpx
from slay import definitions
from truss.remote import remote_factory

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
