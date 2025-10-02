import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional

from truss.cli.logs.utils import ParsedLog, parse_logs
from truss.cli.utils.output import console
from truss.remote.baseten.api import BasetenApi

POLL_INTERVAL_SEC = 2
# NB(nikhil): This helps account for (1) log processing delays (2) clock skews
CLOCK_SKEW_BUFFER_MS = 60000


class LogWatcher(ABC):
    api: BasetenApi
    # NB(nikhil): we add buffer for clock skew, so this helps us detect duplicates.
    # TODO(nikhil): clean up hashes so this doesn't grow indefinitely.
    _log_hashes: set[str] = set()

    _last_poll_time_ms: Optional[int] = None
    _last_log_time_ms: Optional[int] = None

    def __init__(self, api: BasetenApi):
        self.api = api

    def _hash_log(self, log: ParsedLog) -> str:
        log_str = f"{log.timestamp}-{log.message}-{log.replica}"
        return hashlib.sha256(log_str.encode("utf-8")).hexdigest()

    def get_start_epoch_ms(self, now_ms: int) -> Optional[int]:
        if self._last_poll_time_ms:
            return self._last_poll_time_ms - CLOCK_SKEW_BUFFER_MS

        return None

    def fetch_and_parse_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> Iterator[ParsedLog]:
        api_logs = self.fetch_logs(
            start_epoch_millis=start_epoch_millis, end_epoch_millis=end_epoch_millis
        )

        parsed_logs = parse_logs(api_logs)

        for log in parsed_logs:
            if (h := self._hash_log(log)) not in self._log_hashes:
                self._log_hashes.add(h)

                yield log

    def poll(self) -> Iterator[ParsedLog]:
        now_ms = int(time.time() * 1000)
        start_epoch_ms = self.get_start_epoch_ms(now_ms)

        for log in self.fetch_and_parse_logs(
            start_epoch_millis=start_epoch_ms,
            end_epoch_millis=now_ms + CLOCK_SKEW_BUFFER_MS,
        ):
            yield log

            epoch_ns = int(log.timestamp)
            self._last_log_time_ms = int(epoch_ns / 1e6)

        self._last_poll_time_ms = now_ms

    def watch(self) -> Iterator[ParsedLog]:
        self.before_polling()
        with console.status("Polling logs", spinner="aesthetic"):
            while True:
                for log in self.poll():
                    yield log
                if self._log_hashes:
                    break
                time.sleep(POLL_INTERVAL_SEC)

            while self.should_poll_again():
                for log in self.poll():
                    yield log
                time.sleep(POLL_INTERVAL_SEC)
                self.post_poll()

        self.after_polling()

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
    def after_polling(self) -> None:
        """Hook to run code after all polling ends."""
        pass

    @abstractmethod
    def should_poll_again(self) -> bool:
        """Hook that returns whether we should poll again."""
        pass

    @abstractmethod
    def post_poll(self) -> None:
        """Hook to run code after an individual poll."""
        pass
