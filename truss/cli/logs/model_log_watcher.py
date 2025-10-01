from typing import Any, List, Optional, Iterator
import time

from truss.cli.logs.base_watcher import LogWatcher
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.status import MODEL_RUNNING_STATES
from functools import cached_property
from truss.cli.logs.utils import ParsedLog


MAX_LOOK_BACK_MS = 1000 * 60 * 60  # 1 hour.


class ModelDeploymentLogWatcher(LogWatcher):
    _model_id: str
    _deployment_id: str
    _current_status: Optional[str] = None

    def __init__(self, api: BasetenApi, model_id: str, deployment_id: str):
        super().__init__(api)
        self._model_id = model_id
        self._deployment_id = deployment_id

    def before_polling(self) -> None:
        self._current_status = self._get_current_status()

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        return self.api.get_model_deployment_logs(
            self._model_id, self._deployment_id, start_epoch_millis, end_epoch_millis
        )

    def poll(self) -> Iterator[ParsedLog]:
        # NOTE(Tyron): If there can be multiple replicas,
        # we can't use a timestamp cursor to poll for logs.
        if not self._is_development:
            yield from super().poll()

            return

        # Cursor logic. Assumes single replica.

        start_epoch_ms = self._last_poll_time
        now_ms = int(time.time() * 1000)

        if start_epoch_ms:
            start_epoch_ms = max(start_epoch_ms, now_ms - MAX_LOOK_BACK_MS)

        for log in self.fetch_and_parse_logs(
            start_epoch_millis=start_epoch_ms, end_epoch_millis=now_ms
        ):
            yield log

            epoch_ns = int(log.timestamp)
            self._last_poll_time = int(epoch_ns / 1e6)

    def should_poll_again(self) -> bool:
        return self._current_status in MODEL_RUNNING_STATES

    def _get_deployment(self) -> Any:
        return self.api.get_deployment(self._model_id, self._deployment_id)

    def _get_current_status(self) -> str:
        return self._get_deployment()["status"]

    @cached_property
    def _is_development(self) -> bool:
        return self._get_deployment()["is_development"]

    def post_poll(self) -> None:
        self._current_status = self._get_current_status()

    def after_polling(self) -> None:
        pass
