import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import rich

from truss.remote.baseten.api import BasetenApi
from truss.shared.types import ParsedLog

CLOCK_SKEW_BUFFER_MS = 1000
POLL_INTERVAL_SEC = 2


def _parse_logs(api_logs: List[Any]) -> List[ParsedLog]:
    return [ParsedLog.from_raw(api_log) for api_log in api_logs]


def format_and_output_logs(api_logs: List[Any], console: "rich.Console"):
    logs = _parse_logs(api_logs)
    # The API returns the most recent results first, but users expect
    # to see those at the bottom.
    for log in logs[::-1]:
        log.to_console(console)


class LogWatcher(ABC):
    api: BasetenApi
    console: "rich.Console"
    # NB(nikhil): we add buffer for clock skew, so this helps us detect duplicates.
    # TODO(nikhil): clean up hashes so this doesn't grow indefinitely.
    _log_hashes: set[str] = set()
    _last_poll_time: Optional[int] = None

    def __init__(self, api: BasetenApi, console: "rich.Console"):
        self.api = api
        self.console = console

    def _hash_log(self, log: ParsedLog) -> str:
        log_str = f"{log.timestamp}-{log.message}-{log.replica}"
        return hashlib.sha256(log_str.encode("utf-8")).hexdigest()

    def _poll(self) -> None:
        start_epoch: Optional[int] = None
        now = int(time.time() * 1000)
        if self._last_poll_time is not None:
            start_epoch = self._last_poll_time - CLOCK_SKEW_BUFFER_MS

        api_logs = self.fetch_logs(
            start_epoch_millis=start_epoch, end_epoch_millis=now + CLOCK_SKEW_BUFFER_MS
        )

        parsed_logs = _parse_logs(api_logs)
        for log in parsed_logs[::-1]:
            h = self._hash_log(log)
            if h not in self._log_hashes:
                self._log_hashes.add(h)
                log.to_console(self.console)

        self._last_poll_time = now

    def watch(self) -> None:
        self.before_polling()
        with self.console.status("Waiting for logs...", spinner="dots"):
            while True:
                self._poll()
                if self._log_hashes:
                    break
                time.sleep(POLL_INTERVAL_SEC)

        while self.should_poll_again():
            self._poll()
            time.sleep(POLL_INTERVAL_SEC)

            self.post_poll()

    @abstractmethod
    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        pass

    @abstractmethod
    def before_polling(self) -> None:
        """Hook to run code before any polling begins."""
        pass

    @abstractmethod
    def should_poll_again(self) -> bool:
        """Hook that returns whether we should poll again."""
        pass

    @abstractmethod
    def post_poll(self) -> None:
        """Hook to run code after an individual poll."""
        pass
