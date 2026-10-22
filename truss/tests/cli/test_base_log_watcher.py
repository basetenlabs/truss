from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from truss.cli.logs.base_watcher import LogWatcher


class _DummyLogWatcher(LogWatcher):
    def __init__(self, api: Any, batches: List[List[dict]]):
        super().__init__(api)
        self._batches = list(batches)
        self._poll_count = 0

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        if self._batches:
            return self._batches.pop(0)
        return []

    def before_polling(self) -> None:
        pass

    def after_polling(self) -> None:
        pass

    def should_poll_again(self) -> bool:
        return len(self._batches) > 0

    def post_poll(self) -> None:
        self._poll_count += 1


def _raw_log(timestamp: str = "1700000000000000000", message: str = "hello", replica: str = "r1") -> dict:
    return {"timestamp": timestamp, "message": message, "replica": replica}


def test_log_watcher_dedup_hashes_are_per_instance():
    entry = _raw_log()
    watcher1 = _DummyLogWatcher(MagicMock(), [[entry]])
    watcher2 = _DummyLogWatcher(MagicMock(), [[entry]])

    logs1 = list(watcher1.poll())
    logs2 = list(watcher2.poll())

    assert len(logs1) == 1
    assert len(logs2) == 1
    assert watcher1._log_hashes is not watcher2._log_hashes


@patch("truss.cli.logs.base_watcher.time.sleep")
def test_second_watcher_still_waits_for_first_log(mock_sleep):
    first = _DummyLogWatcher(MagicMock(), [[_raw_log(message="first")]])
    assert [log.message for log in first.watch(show_spinner=False)] == ["first"]

    second = _DummyLogWatcher(MagicMock(), [[], [_raw_log(message="late")]])
    assert [log.message for log in second.watch(show_spinner=False)] == ["late"]
    assert mock_sleep.called
