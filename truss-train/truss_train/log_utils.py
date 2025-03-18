import hashlib
import time
from datetime import datetime
from typing import Any, List, Optional

import pydantic
import rich
from rich import text

from truss.remote.baseten.api import BasetenApi

POLL_INTERVAL_SEC = 2
CLOCK_SKEW_BUFFER_MS = 1000

# NB(nikhil): When a job ends, we poll for this many seconds after to capture
# any trailing logs that contain information about errors.
JOB_TERMINATION_GRACE_PERIOD_SEC = 10

JOB_STARTING_STATES = ["TRAINING_JOB_CREATED", "TRAINING_JOB_DEPLOYING"]
JOB_RUNNING_STATES = ["TRAINING_JOB_RUNNING"]


class RawTrainingJobLog(pydantic.BaseModel):
    timestamp: str
    message: str
    replica: Optional[str]


def _output_log(log: RawTrainingJobLog, console: "rich.Console"):
    epoch_nanos = int(log.timestamp)
    dt = datetime.fromtimestamp(epoch_nanos / 1e9)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    time_text = text.Text(f"[{formatted_time}]", style="blue")

    assert log.replica is not None
    replica_text = text.Text(f" ({log.replica})", style="green")
    message_text = text.Text(f": {log.message.strip()}", style="white")

    # Output the combined log line to the console
    console.print(time_text, replica_text, message_text, sep="")


def _parse_logs(api_logs: List[Any]) -> List[RawTrainingJobLog]:
    return [RawTrainingJobLog(**log) for log in api_logs]


def format_and_output_logs(api_logs: List[Any], console: "rich.Console"):
    logs = _parse_logs(api_logs)
    # The API returns the most recent results first, but users expect
    # to see those at the bottom.
    for log in logs[::-1]:
        _output_log(log, console)


class LogWatcher:
    api: BasetenApi
    project_id: str
    job_id: str
    console: "rich.Console"
    # NB(nikhil): we add buffer for clock skew, so this helps us detect duplicates.
    # TODO(nikhil): clean up hashes so this doesn't grow indefinitely.
    _log_hashes: set[str] = set()
    _last_poll_time: Optional[int] = None
    _poll_stop_time: Optional[int] = None

    def __init__(
        self, api: BasetenApi, project_id: str, job_id: str, console: "rich.Console"
    ):
        self.api = api
        self.project_id = project_id
        self.job_id = job_id
        self.console = console

    def _hash_log(self, log: RawTrainingJobLog) -> str:
        log_str = f"{log.timestamp}-{log.message}-{log.replica}"
        return hashlib.sha256(log_str.encode("utf-8")).hexdigest()

    def _poll(self) -> None:
        start_epoch: Optional[int] = None
        now = int(time.time() * 1000)
        if self._last_poll_time is not None:
            start_epoch = self._last_poll_time - CLOCK_SKEW_BUFFER_MS

        api_logs = self.api.get_training_job_logs(
            project_id=self.project_id,
            job_id=self.job_id,
            start_epoch_millis=start_epoch,
            end_epoch_millis=now + CLOCK_SKEW_BUFFER_MS,
        )

        parsed_logs = _parse_logs(api_logs)
        for log in parsed_logs[::-1]:
            h = self._hash_log(log)
            if h not in self._log_hashes:
                self._log_hashes.add(h)
                _output_log(log, self.console)

        self._last_poll_time = now

    def _get_current_job_status(self) -> str:
        job = self.api.get_training_job(self.project_id, self.job_id)
        return job["current_status"]

    def _wait_until_running(self) -> None:
        current_status = self._get_current_job_status()
        status_str = "Waiting for job to run, currently {current_status}..."
        with self.console.status(
            status_str.format(current_status=current_status), spinner="dots"
        ) as status_console:
            while current_status in JOB_STARTING_STATES:
                time.sleep(POLL_INTERVAL_SEC)
                current_status = self._get_current_job_status()
                status_console.update(status_str.format(current_status=current_status))

    def _poll_final_logs(self):
        if self._poll_stop_time is None:
            return False

        return int(time.time()) <= self._poll_stop_time

    def _maybe_update_poll_stop_time(self, current_status: str) -> None:
        if current_status not in JOB_RUNNING_STATES and self._poll_stop_time is None:
            self._poll_stop_time = int(time.time()) + JOB_TERMINATION_GRACE_PERIOD_SEC

    def watch(self) -> None:
        self._wait_until_running()
        with self.console.status("Waiting for logs...", spinner="dots"):
            while True:
                self._poll()
                if self._log_hashes:
                    break
                time.sleep(POLL_INTERVAL_SEC)

        current_status = self._get_current_job_status()
        while current_status in JOB_RUNNING_STATES or self._poll_final_logs():
            self._poll()
            time.sleep(POLL_INTERVAL_SEC)

            current_status = self._get_current_job_status()
            self._maybe_update_poll_stop_time(current_status)
