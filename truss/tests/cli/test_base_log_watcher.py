from typing import Any, Dict, List, Optional

import pytest

from truss.cli.logs import base_watcher
from truss.cli.logs.model_log_watcher import ModelDeploymentLogWatcher

# Upper bounds that turn an accidental infinite polling loop into a test failure.
MAX_SLEEPS = 50
MAX_FETCHES = 50


def _raw_log(message: str, timestamp: str = "1700000000000000000") -> Dict[str, Any]:
    return {"timestamp": timestamp, "message": message, "replica": "r0"}


class FakeApi:
    """Stands in for BasetenApi; no network.

    ``logs_by_fetch`` holds the logs returned by the Nth fetch (0-indexed); fetches
    past the end return the last entry. The deployment reports ``DEPLOYING`` for
    ``running_status_checks`` status lookups and ``FAILED`` afterwards, which ends
    ``watch()``.
    """

    def __init__(
        self, logs_by_fetch: List[List[Dict[str, Any]]], running_status_checks: int = 2
    ):
        self._logs_by_fetch = logs_by_fetch
        self._running_status_checks = running_status_checks
        self.fetches = 0
        self.status_checks = 0

    def get_model_deployment_logs(
        self,
        model_id: str,
        deployment_id: str,
        start_epoch_millis: Optional[int],
        end_epoch_millis: Optional[int],
    ) -> List[Dict[str, Any]]:
        if self.fetches >= MAX_FETCHES:
            raise AssertionError("watcher kept fetching; it would have hung")
        logs = self._logs_by_fetch[min(self.fetches, len(self._logs_by_fetch) - 1)]
        self.fetches += 1
        return logs

    def get_deployment(self, model_id: str, deployment_id: str) -> Dict[str, Any]:
        self.status_checks += 1
        running = self.status_checks <= self._running_status_checks
        return {"status": "DEPLOYING" if running else "FAILED", "is_development": False}


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    """Replace time.sleep in the watcher with a recorder that cannot block.

    Raises once MAX_SLEEPS is exceeded, so a loop that never terminates fails fast.
    """
    calls: List[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)
        if len(calls) > MAX_SLEEPS:
            raise AssertionError("watcher kept sleeping; it would have hung")

    monkeypatch.setattr(base_watcher.time, "sleep", fake_sleep)
    return calls


def _watch_all(api: FakeApi, deployment_id: str = "d") -> List[str]:
    watcher = ModelDeploymentLogWatcher(api, "m", deployment_id)  # type: ignore[arg-type]
    return [log.message for log in watcher.watch(show_spinner=False)]


def test_each_watcher_has_its_own_dedup_state(sleeps):
    first_api = FakeApi([[_raw_log("boot")]])
    second_api = FakeApi([[_raw_log("boot")]])
    first = ModelDeploymentLogWatcher(first_api, "m", "d1")  # type: ignore[arg-type]
    second = ModelDeploymentLogWatcher(second_api, "m", "d2")  # type: ignore[arg-type]

    assert [log.message for log in first.watch(show_spinner=False)] == ["boot"]
    # Identical content on a different deployment is not a duplicate.
    assert [log.message for log in second.watch(show_spinner=False)] == ["boot"]
    assert first._log_hashes is not second._log_hashes
    assert len(first._log_hashes) == 1
    assert len(second._log_hashes) == 1


def test_fresh_watcher_waits_for_late_logs_after_another_watcher_ran(sleeps):
    assert _watch_all(FakeApi([[_raw_log("boot")]]), "d1") == ["boot"]

    # No logs for the first three fetches, then one arrives. A watcher that
    # inherited the previous watcher's hashes would skip the wait-for-first-log
    # loop and exit before seeing it.
    late_api = FakeApi([[], [], [], [_raw_log("late", "1700000001000000000")]])
    assert _watch_all(late_api, "d2") == ["late"]
    assert late_api.fetches > 3


def test_repeated_polls_dedup_within_one_watcher(sleeps):
    # The clock-skew buffer makes consecutive polls re-fetch old logs.
    first = _raw_log("one", "1700000000000000000")
    second = _raw_log("two", "1700000001000000000")
    api = FakeApi([[first], [first, second], [first, second]])
    watcher = ModelDeploymentLogWatcher(api, "m", "d")  # type: ignore[arg-type]

    assert [log.message for log in watcher.poll()] == ["one"]
    assert [log.message for log in watcher.poll()] == ["two"]
    assert [log.message for log in watcher.poll()] == []
