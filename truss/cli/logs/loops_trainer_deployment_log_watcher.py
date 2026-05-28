from typing import Any, List, Optional

from truss.cli.logs.base_watcher import LogWatcher
from truss.remote.baseten.api import BasetenApi

# TrainerDeployment statuses that are still producing (or about to produce)
# logs. Tail mode keeps polling while the deployment is in one of these
# states; once it transitions to FAILED or STOPPED, the watcher exits.
_RUNNING_STATUSES = {"CREATED", "DEPLOYING", "RUNNING"}


class LoopsTrainerDeploymentLogWatcher(LogWatcher):
    """Tails logs from a Loops trainer deployment's pods.

    Mirrors ``TrainingLogWatcher`` — same poll-with-clock-skew-buffer pattern
    from ``LogWatcher`` — but uses the trainer-deployment-scoped logs endpoint
    and exits when the deployment is no longer in a running state.
    """

    _trainer_deployment_id: str
    _current_status: Optional[str] = None

    def __init__(self, api: BasetenApi, trainer_deployment_id: str):
        super().__init__(api)
        self._trainer_deployment_id = trainer_deployment_id

    def before_polling(self) -> None:
        self._current_status = self._get_current_status()

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        return self.api.get_loops_deployment_logs(
            self._trainer_deployment_id, start_epoch_millis, end_epoch_millis
        )

    def should_poll_again(self) -> bool:
        return self._current_status in _RUNNING_STATUSES

    def post_poll(self) -> None:
        self._current_status = self._get_current_status()

    def after_polling(self) -> None:
        pass

    def _get_current_status(self) -> Optional[str]:
        # The Loops deployments list view returns each deployment's latest
        # status. Filter for ours; if the backend stops returning it (e.g.
        # post-deactivation) treat that as a terminal signal so tail exits.
        deployments = self.api.list_loops_deployments()
        for deployment in deployments:
            if deployment.get("id") == self._trainer_deployment_id:
                status = deployment.get("status") or {}
                return status.get("name")
        return None
