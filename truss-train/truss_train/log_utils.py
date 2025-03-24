import time
from typing import Any, List, Optional

from truss.remote.baseten.api import BasetenApi
from truss.shared.log_watcher import POLL_INTERVAL_SEC, LogWatcher
from truss.shared.types import SpinnerFactory

# NB(nikhil): When a job ends, we poll for this many seconds after to capture
# any trailing logs that contain information about errors.
JOB_TERMINATION_GRACE_PERIOD_SEC = 10

JOB_STARTING_STATES = ["TRAINING_JOB_CREATED", "TRAINING_JOB_DEPLOYING"]
JOB_RUNNING_STATES = ["TRAINING_JOB_RUNNING"]


class TrainingLogWatcher(LogWatcher):
    project_id: str
    job_id: str
    _poll_stop_time: Optional[int] = None
    _current_status: Optional[str] = None

    def __init__(
        self,
        api: BasetenApi,
        project_id: str,
        job_id: str,
        spinner_factory: SpinnerFactory,
    ):
        super().__init__(api, spinner_factory)
        self.project_id = project_id
        self.job_id = job_id

    def _get_current_job_status(self) -> str:
        job = self.api.get_training_job(self.project_id, self.job_id)
        return job["current_status"]

    def before_polling(self) -> None:
        self._current_status = self._get_current_job_status()
        status_str = "Waiting for job to run, currently {current_status}..."
        with self.spinner_factory(
            status_str.format(current_status=self._current_status)
        ) as spinner:
            while self._current_status in JOB_STARTING_STATES:
                time.sleep(POLL_INTERVAL_SEC)
                self._current_status = self._get_current_job_status()
                spinner.update(status_str.format(current_status=self._current_status))

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        return self.api.get_training_job_logs(
            self.project_id, self.job_id, start_epoch_millis, end_epoch_millis
        )

    def should_poll_again(self) -> bool:
        return self._current_status in JOB_RUNNING_STATES or self._poll_final_logs()

    def post_poll(self) -> None:
        self._current_status = self._get_current_job_status()
        self._maybe_update_poll_stop_time(self._current_status)

    def _poll_final_logs(self):
        if self._poll_stop_time is None:
            return False

        return int(time.time()) <= self._poll_stop_time

    def _maybe_update_poll_stop_time(self, current_status: str) -> None:
        if current_status not in JOB_RUNNING_STATES and self._poll_stop_time is None:
            self._poll_stop_time = int(time.time()) + JOB_TERMINATION_GRACE_PERIOD_SEC
