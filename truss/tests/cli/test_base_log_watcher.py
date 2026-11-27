from typing import Any, List, Optional

import pytest

from truss.cli.logs.base_watcher import LogWatcher


def _raw_log(message: str) -> dict:
    # timestamp is epoch nanos as a string, matching what the API returns.
    return {
        "timestamp": "1700000000000000000",
        "message": message,
        "replica": "replica-0",
    }


class FakeLogWatcher(LogWatcher):
    """A LogWatcher backed by canned API responses (no network).

    `responses` is a queue: each `fetch_logs` call pops the next response,
    and repeats the last one once the queue is exhausted.
    """

    def __init__(self, responses: List[List[dict]]):
        super().__init__(api=None)  # type: ignore[arg-type]
        self._responses = responses
        self._fetch_calls = 0

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        index = min(self._fetch_calls, len(self._responses) - 1)
        self._fetch_calls += 1
        return self._responses[index]

    def before_polling(self) -> None:
        pass

    def after_polling(self) -> None:
        pass

    def should_poll_again(self) -> bool:
        return False

    def post_poll(self) -> None:
        pass


def test_log_hashes_are_per_instance():
    watcher_a = FakeLogWatcher([[_raw_log("hello")]])
    watcher_b = FakeLogWatcher([[_raw_log("hello")]])
    assert watcher_a._log_hashes is not watcher_b._log_hashes
    assert "_log_hashes" not in LogWatcher.__dict__


def test_second_watcher_emits_identical_logs():
    watcher_a = FakeLogWatcher([[_raw_log("hello")]])
    watcher_b = FakeLogWatcher([[_raw_log("hello")]])
    assert [log.message for log in watcher_a.poll()] == ["hello"]
    # Fails before the fix: the shared class-level set already holds the hash,
    # so the second watcher drops the identical log.
    assert [log.message for log in watcher_b.poll()] == ["hello"]


def test_repeated_polls_dedup_within_single_watcher():
    watcher = FakeLogWatcher([[_raw_log("hello")]])
    assert [log.message for log in watcher.poll()] == ["hello"]
    assert [log.message for log in watcher.poll()] == []


def test_second_watcher_waits_for_late_logs(monkeypatch: pytest.MonkeyPatch):
    # First watcher runs to completion, populating its own dedup set.
    first = FakeLogWatcher([[_raw_log("hello")]])
    assert [log.message for log in first.watch(show_spinner=False)] == ["hello"]

    # Second watcher sees no logs on its first poll, then a late log.
    # Fails before the fix: `watch()` breaks out of its "wait for first log"
    # loop on `if self._log_hashes:` because the class-level set is already
    # non-empty, so the late log is never seen.
    monkeypatch.setattr("truss.cli.logs.base_watcher.time.sleep", lambda _: None)
    second = FakeLogWatcher([[], [_raw_log("late")]])
    assert [log.message for log in second.watch(show_spinner=False)] == ["late"]
