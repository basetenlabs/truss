from typing import Any, List, Optional

import rich

from truss.remote.baseten.api import BasetenApi
from truss.shared.log_watcher import LogWatcher

# NB(nikhil): These are slightly translated verisons of our internal model state machine.
MODEL_RUNNING_STATES = [
    "BUILDING",
    "DEPLOYING",
    "LOADING_MODEL",
    "ACTIVE",
    "UPDATING",
    "WAKING_UP",
]


class ModelDeploymentLogWatcher(LogWatcher):
    _model_id: str
    _deployment_id: str
    _current_status: Optional[str] = None

    def __init__(
        self,
        api: BasetenApi,
        model_id: str,
        deployment_id: str,
        console: "rich.Console",
    ):
        super().__init__(api, console)
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

    def should_poll_again(self) -> bool:
        return self._current_status in MODEL_RUNNING_STATES

    def _get_current_status(self) -> str:
        return self.api.get_deployment(self._model_id, self._deployment_id)["status"]

    def post_poll(self) -> None:
        self._current_status = self._get_current_status()
