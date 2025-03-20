import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional

from truss.remote.baseten.api import BasetenApi
from truss.shared.types import ParsedLog, SpinnerFactory

CLOCK_SKEW_BUFFER_MS = 1000
POLL_INTERVAL_SEC = 2


def parse_logs(api_logs: List[Any]) -> List[ParsedLog]:
    return [ParsedLog.from_raw(api_log) for api_log in api_logs]


class LogWatcher(ABC):
    api: BasetenApi
    spinner_factory: SpinnerFactory
    # NB(nikhil): we add buffer for clock skew, so this helps us detect duplicates.
    # TODO(nikhil): clean up hashes so this doesn't grow indefinitely.
    _log_hashes: set[str] = set()
    _last_poll_time: Optional[int] = None

    def __init__(self, api: BasetenApi, spinner_factory: SpinnerFactory):
        self.api = api
        self.spinner_factory = spinner_factory

    def _hash_log(self, log: ParsedLog) -> str:
        log_str = f"{log.timestamp}-{log.message}-{log.replica}"
        return hashlib.sha256(log_str.encode("utf-8")).hexdigest()

    def _poll(self) -> Iterator[ParsedLog]:
        start_epoch: Optional[int] = None
        now = int(time.time() * 1000)
        if self._last_poll_time is not None:
            start_epoch = self._last_poll_time - CLOCK_SKEW_BUFFER_MS

        api_logs = self.fetch_logs(
            start_epoch_millis=start_epoch, end_epoch_millis=now + CLOCK_SKEW_BUFFER_MS
        )

        parsed_logs = parse_logs(api_logs)
        for log in parsed_logs[::-1]:
            h = self._hash_log(log)
            if h not in self._log_hashes:
                self._log_hashes.add(h)
                yield log

        self._last_poll_time = now

    def watch(self) -> Iterator[ParsedLog]:
        self.before_polling()
        with self.spinner_factory("Waiting for logs..."):
            while True:
                for log in self._poll():
                    yield log
                if self._log_hashes:
                    break
                time.sleep(POLL_INTERVAL_SEC)

        while self.should_poll_again():
            for log in self._poll():
                yield log
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
