from typing import Any
from unittest.mock import Mock

import pytest
from truss.templates.server.common.retry import retry


class FailForCallCount:
    def __init__(self, count: int) -> None:
        self._fail_for_call_count = count
        self._call_count = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._call_count += 1
        if self._call_count <= self._fail_for_call_count:
            raise RuntimeError()

    @property
    def call_count(self) -> int:
        return self._call_count


def fail_for_call_count(count: int) -> callable:
    call_count = 0

    def inner():
        nonlocal call_count

    return inner


def test_no_fail_no_retry():
    mock = Mock()
    mock_logging_fn = Mock()
    retry(mock, 3, mock_logging_fn, "")
    mock.assert_called_once()
    mock_logging_fn.assert_not_called()


def test_retry_fail_once():
    mock_logging_fn = Mock()
    fail_mock = FailForCallCount(1)
    retry(fail_mock, 3, mock_logging_fn, "Failed")
    assert fail_mock.call_count == 2
    assert mock_logging_fn.call_count == 1
    mock_logging_fn.assert_called_once_with("Failed Retrying...")


def test_retry_fail_twice():
    mock_logging_fn = Mock()
    fail_mock = FailForCallCount(2)
    retry(fail_mock, 3, mock_logging_fn, "Failed")
    assert fail_mock.call_count == 3
    assert mock_logging_fn.call_count == 2
    mock_logging_fn.assert_called_with("Failed Retrying. Retry count: 1")


def test_retry_fail_twice_gap():
    mock_logging_fn = Mock()
    fail_mock = FailForCallCount(2)
    retry(fail_mock, 3, mock_logging_fn, "Failed", gap_seconds=0.1)
    assert fail_mock.call_count == 3
    assert mock_logging_fn.call_count == 2
    mock_logging_fn.assert_called_with("Failed Retrying. Retry count: 1")


def test_retry_fail_more_than_limit():
    mock_logging_fn = Mock()
    fail_mock = FailForCallCount(3)
    with pytest.raises(RuntimeError):
        retry(fail_mock, 1, mock_logging_fn, "Failed")
    assert fail_mock.call_count == 2
    assert mock_logging_fn.call_count == 1
