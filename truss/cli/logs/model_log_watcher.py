import signal
import time
from typing import Any, Iterator, List, Optional

from truss.cli.logs.base_watcher import LogWatcher
from truss.cli.logs.utils import ParsedLog
from truss.cli.utils.output import console
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.status import MODEL_RUNNING_STATES

POLL_INTERVAL_SEC = 2
CLOCK_SKEW_BUFFER_MS = 60000


class ModelDeploymentLogWatcher(LogWatcher):
    _model_id: str
    _deployment_id: str
    _current_status: Optional[str] = None
    _tail_only: bool = False

    def __init__(
        self,
        api: BasetenApi,
        model_id: str,
        deployment_id: str,
        tail_only: bool = False,
        start_epoch_millis: Optional[int] = None,
    ):
        super().__init__(api)
        self._model_id = model_id
        self._deployment_id = deployment_id
        self._tail_only = tail_only
        self._start_epoch_millis = start_epoch_millis
        # Register SIGINT handler to show helpful message on Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        msg = f"\n\nExiting deployment logs. To continue viewing logs, run `truss deployment logs --tail --deployment-id {self._deployment_id}`"
        console.print(msg, style="yellow")
        raise KeyboardInterrupt()

    def before_polling(self) -> None:
        self._current_status = self._get_current_status()

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        return self.api.get_model_deployment_logs(
            self._model_id, self._deployment_id, start_epoch_millis, end_epoch_millis
        )

    def should_poll_again(self) -> bool:
        return self._current_status in MODEL_RUNNING_STATES

    def _get_current_status(self) -> str:
        return self.api.get_deployment(self._model_id, self._deployment_id)["status"]

    def post_poll(self) -> None:
        self._current_status = self._get_current_status()

    def after_polling(self) -> None:
        pass

    def watch(self) -> Iterator[ParsedLog]:
        """Override the base watch method to support tail-only mode."""
        self.before_polling()

        if self._tail_only:
            # For tail-only mode, start from specified time or current time
            if self._start_epoch_millis is not None:
                self._last_poll_time = self._start_epoch_millis
                # First, show historical logs from the specified start time
                with console.status("Fetching historical logs", spinner="aesthetic"):
                    for log in self._poll():
                        yield log
            else:
                self._last_poll_time = int(time.time() * 1000)

            # Then continue tailing if deployment is running
            with console.status("Tailing deployment logs", spinner="aesthetic"):
                while self.should_poll_again():
                    for log in self._poll():
                        yield log
                    time.sleep(POLL_INTERVAL_SEC)
                    self.post_poll()
        else:
            # Use the original behavior for backward compatibility
            with console.status("Polling logs", spinner="aesthetic"):
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

        self.after_polling()
