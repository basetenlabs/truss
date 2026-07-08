from typing import Any, List, Optional

from truss.cli.logs.base_watcher import LogWatcher
from truss.remote.baseten.api import BasetenApi

# Loops-deployment statuses that are still producing (or about to produce)
# logs. Tail mode keeps polling while the deployment is in one of these
# states; once it transitions to FAILED or STOPPED, the watcher exits.
_RUNNING_STATUSES = {"CREATED", "DEPLOYING", "RUNNING"}


class LoopsDeploymentLogWatcher(LogWatcher):
    """Tails logs from a Loops deployment.

    Mirrors ``TrainingLogWatcher`` — same poll-with-clock-skew-buffer pattern
    from ``LogWatcher`` — but uses the Loops-deployment-scoped logs endpoint
    and exits when the deployment is no longer in a running state.
    """

    _loops_deployment_id: str
    _current_status: Optional[str] = None

    def __init__(self, api: BasetenApi, loops_deployment_id: str):
        super().__init__(api)
        self._loops_deployment_id = loops_deployment_id

    def before_polling(self) -> None:
        self._current_status = self._get_current_status()

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        # Deliberately the single-page fetch: tail polls re-fetch
        # overlapping windows every few seconds and want the newest lines
        # of each, not a multi-request crawl of the whole window.
        return self.api.get_loops_deployment_logs_page(
            self._loops_deployment_id, start_epoch_millis, end_epoch_millis
        )

    def should_poll_again(self) -> bool:
        return self._current_status in _RUNNING_STATUSES

    def post_poll(self) -> None:
        self._current_status = self._get_current_status()

    def after_polling(self) -> None:
        pass

    def _get_current_status(self) -> Optional[str]:
        # Hit the per-deployment endpoint instead of paging the full list —
        # cheaper and the response carries the latest status directly. A 404
        # (e.g. post-deactivation) surfaces as an exception we let propagate;
        # the watcher will exit on the next ``should_poll_again`` call.
        deployment = self.api.get_loops_deployment(self._loops_deployment_id)
        status = deployment.get("status") or {}
        return status.get("name")
