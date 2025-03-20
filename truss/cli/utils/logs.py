from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, List, Optional

from rich import console as rich_console
from rich import text

from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.status import MODEL_RUNNING_STATES
from truss.shared.log_watcher import LogWatcher
from truss.shared.types import ParsedLog, Spinner, SpinnerFactory


def gen_spinner_factory(console: rich_console.Console) -> SpinnerFactory:
    @contextmanager
    def spinner_factory(text: str) -> Iterator[Spinner]:
        with console.status(text, spinner="dots") as status:
            yield status

    return spinner_factory


def output_log(log: ParsedLog, console: rich_console.Console):
    epoch_nanos = int(log.timestamp)
    dt = datetime.fromtimestamp(epoch_nanos / 1e9)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    time_text = text.Text(f"[{formatted_time}]: ", style="blue")
    message_text = text.Text(f"{log.message.strip()}", style="white")
    if log.replica:
        replica_text = text.Text(f"({log.replica}) ", style="green")
    else:
        replica_text = text.Text()

    # Output the combined log line to the console
    console.print(time_text, replica_text, message_text, sep="")


class ModelDeploymentLogWatcher(LogWatcher):
    _model_id: str
    _deployment_id: str
    _current_status: Optional[str] = None

    def __init__(
        self,
        api: BasetenApi,
        model_id: str,
        deployment_id: str,
        spinner_factory: SpinnerFactory,
    ):
        super().__init__(api, spinner_factory)
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
