import time
from dataclasses import dataclass
from typing import Optional

from truss.cli.utils.output import console
from truss.remote.baseten.api import BasetenApi

POLL_INTERVAL_SEC = 2

# NB(nikhil): When a job ends, we poll for this many seconds after to capture
# any trailing logs that contain information about errors.
JOB_TERMINATION_GRACE_PERIOD_SEC = 10
JOB_STARTING_STATES = ["TRAINING_JOB_CREATED", "TRAINING_JOB_DEPLOYING"]
JOB_RUNNING_STATES = ["TRAINING_JOB_RUNNING"]
STATES_WITH_ERROR_MESSAGES = ["TRAINING_JOB_DEPLOY_FAILED"]


@dataclass
class Status:
    status: str
    error_message: Optional[str]


class TrainingPollerMixin:
    api: BasetenApi
    project_id: str
    job_id: str
    _current_status: Status
    _poll_stop_time: Optional[int]

    def __init__(self, api: BasetenApi, project_id: str, job_id: str):
        self.api = api
        self.project_id = project_id
        self.job_id = job_id
        self._current_status = Status(status="", error_message=None)
        self._poll_stop_time = None

    def before_polling(self) -> None:
        self._update_from_current_status()
        status_str = "Waiting for job to run, currently {current_status}..."
        with console.status(
            status_str.format(current_status=self._current_status.status),
            spinner="dots",
        ) as status:
            while self._current_status.status in JOB_STARTING_STATES:
                time.sleep(POLL_INTERVAL_SEC)
                self._update_from_current_status()
                status.update(
                    status_str.format(current_status=self._current_status.status)
                )

    def post_poll(self) -> None:
        self._update_from_current_status()
        self._maybe_update_poll_stop_time()

    def _maybe_update_poll_stop_time(self) -> None:
        if (
            self._current_status.status not in JOB_RUNNING_STATES
            and self._poll_stop_time is None
        ):
            self._poll_stop_time = int(time.time()) + JOB_TERMINATION_GRACE_PERIOD_SEC

    def should_poll_again(self) -> bool:
        return bool(
            self._current_status.status in JOB_RUNNING_STATES
            or self._do_cleanup_polling()
        )

    def _do_cleanup_polling(self):
        if self._poll_stop_time is None:
            return False

        return int(time.time()) <= self._poll_stop_time

    def after_polling(self) -> None:
        if (
            self._current_status.status in STATES_WITH_ERROR_MESSAGES
            and self._current_status.error_message
        ):
            console.print(self._current_status.error_message, style="red")

        if self._current_status.status == "TRAINING_JOB_COMPLETED":
            console.print("Training job completed successfully.", style="green")
        elif self._current_status.status == "TRAINING_JOB_FAILED":
            console.print("Training job failed during execution.", style="red")
        elif self._current_status.status == "TRAINING_JOB_STOPPED":
            console.print("Training job stopped by user.", style="yellow")
        elif self._current_status.status == "TRAINING_JOB_DEPLOY_FAILED":
            console.print("Training job failed during deployment.", style="red")

    def _update_from_current_status(self) -> None:
        current_job = self.api.get_training_job(self.project_id, self.job_id)
        self._current_status = Status(
            status=current_job["training_job"]["current_status"],
            error_message=current_job["training_job"].get("error_message"),
        )
