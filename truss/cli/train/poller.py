import time
from typing import Optional

from rich.console import Console

from truss.cli.common import POLL_INTERVAL_SEC
from truss.remote.baseten.api import BasetenApi

# NB(nikhil): When a job ends, we poll for this many seconds after to capture
# any trailing logs that contain information about errors.
JOB_TERMINATION_GRACE_PERIOD_SEC = 10
JOB_STARTING_STATES = ["TRAINING_JOB_CREATED", "TRAINING_JOB_DEPLOYING"]
JOB_RUNNING_STATES = ["TRAINING_JOB_RUNNING"]


class TrainingPollerMixin:
    api: BasetenApi
    project_id: str
    job_id: str
    console: Console
    _current_status: Optional[str]
    _poll_stop_time: Optional[int]

    def __init__(self, api: BasetenApi, project_id: str, job_id: str, console: Console):
        self.api = api
        self.project_id = project_id
        self.job_id = job_id
        self.console = console
        self._current_status = None
        self._poll_stop_time = None

    def before_polling(self) -> None:
        self._current_status = self._get_current_job_status()
        status_str = "Waiting for job to run, currently {current_status}..."
        with self.console.status(
            status_str.format(current_status=self._current_status), spinner="dots"
        ) as status:
            while self._current_status in JOB_STARTING_STATES:
                time.sleep(POLL_INTERVAL_SEC)
                self._current_status = self._get_current_job_status()
                status.update(status_str.format(current_status=self._current_status))

    def post_poll(self) -> None:
        self._current_status = self._get_current_job_status()
        self._maybe_update_poll_stop_time(self._current_status)

    def _maybe_update_poll_stop_time(self, current_status: str) -> None:
        if current_status not in JOB_RUNNING_STATES and self._poll_stop_time is None:
            self._poll_stop_time = int(time.time()) + JOB_TERMINATION_GRACE_PERIOD_SEC

    def should_poll_again(self) -> bool:
        return self._current_status in JOB_RUNNING_STATES or self._do_cleanup_polling()

    def _do_cleanup_polling(self):
        if self._poll_stop_time is None:
            return False

        return int(time.time()) <= self._poll_stop_time

    def after_polling(self) -> None:
        if self._current_status == "TRAINING_JOB_COMPLETED":
            self.console.print("Training job completed successfully.", style="green")
        elif self._current_status == "TRAINING_JOB_FAILED":
            self.console.print("Training job failed.", style="red")
        elif self._current_status == "TRAINING_JOB_STOPPED":
            self.console.print("Training job stopped by user.", style="yellow")

    def _get_current_job_status(self) -> str:
        job = self.api.get_training_job(self.project_id, self.job_id)
        return job["training_job"]["current_status"]
