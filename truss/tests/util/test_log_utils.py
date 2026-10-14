import logging
import threading
from typing import Iterator

import pytest

from truss.util.log_utils import LogInterceptor


@pytest.fixture
def root_logger() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.setLevel(logging.INFO)
    yield root
    root.handlers = original_handlers
    root.setLevel(original_level)


def _handler_with_format(fmt: str) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def test_intercepts_logs_without_root_handlers(root_logger):
    root_logger.handlers = []
    with LogInterceptor() as interceptor:
        logging.info("hello")
    assert interceptor.get_logs() == ["hello"]


def test_uses_formatter_of_first_root_handler(root_logger):
    root_logger.handlers = [
        _handler_with_format("CUSTOM %(message)s"),
        _handler_with_format("OTHER %(message)s"),
    ]
    with LogInterceptor() as interceptor:
        logging.info("hello")
    assert interceptor.get_logs() == ["CUSTOM hello"]


def test_root_handler_without_formatter_falls_back_to_message(root_logger):
    root_logger.handlers = [logging.StreamHandler()]
    with LogInterceptor() as interceptor:
        logging.info("hello")
    assert interceptor.get_logs() == ["hello"]


def test_restores_root_handlers_on_exit(root_logger):
    handler = _handler_with_format("%(message)s")
    root_logger.handlers = [handler]
    with LogInterceptor():
        assert logging.getLogger().handlers != [handler]
    assert logging.getLogger().handlers == [handler]


def test_each_thread_collects_only_its_own_logs(root_logger):
    root_logger.handlers = []
    results: dict[str, list[str]] = {}
    ready = threading.Barrier(2)

    def worker(name: str) -> None:
        with LogInterceptor() as interceptor:
            ready.wait(timeout=5)
            logging.info("from %s", name)
            ready.wait(timeout=5)
        results[name] = interceptor.get_logs()

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"a": ["from a"], "b": ["from b"]}
