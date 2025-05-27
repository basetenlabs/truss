import contextlib
import json
import tempfile
import textwrap
from pathlib import Path
from typing import Iterator, Optional

from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle.truss_handle import TrussHandle


def _create_truss(truss_dir: Path, config_contents: str, model_contents: str):
    truss_dir.mkdir(exist_ok=True)  # Ensure the 'truss' directory exists
    truss_model_dir = truss_dir / "model"
    truss_model_dir.mkdir(parents=True, exist_ok=True)

    config_file = truss_dir / "config.yaml"
    model_file = truss_model_dir / "model.py"
    with open(config_file, "w", encoding="utf-8") as file:
        file.write(config_contents)
    with open(model_file, "w", encoding="utf-8") as file:
        file.write(model_contents)


@contextlib.contextmanager
def temp_truss(model_src: str, config_src: str = "") -> Iterator[TrussHandle]:
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        _create_truss(truss_dir, config_src, textwrap.dedent(model_src))
        yield TrussHandle(truss_dir)


DEFAULT_LOG_ERROR = "Internal Server Error"


def _log_contains_line(
    line: dict, message: str, level: str, error: Optional[str] = None
):
    return (
        line["levelname"] == level
        and message in line["message"]
        and (error is None or error in line["exc_info"])
    )


def assert_logs_contain_error(
    logs: str, error: Optional[str], message=DEFAULT_LOG_ERROR
):
    loglines = [json.loads(line) for line in logs.splitlines()]
    assert any(
        _log_contains_line(line, message, "ERROR", error) for line in loglines
    ), (
        f"Did not find expected error in logs.\nExpected error: {error}\n"
        f"Expected message: {message}\nActual logs:\n{loglines}"
    )


def assert_logs_contain(logs: str, message: str, level: str = "INFO"):
    loglines = [json.loads(line) for line in logs.splitlines()]
    assert any(_log_contains_line(line, message, level) for line in loglines), (
        f"Did not find expected  logs.\n"
        f"Expected message: {message}\nActual logs:\n{loglines}"
    )
