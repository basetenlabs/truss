"""Follow-up command vocabulary for truss vs the baseten CLI.

When baseten-cli shells out to truss, it sets BASETEN_TRUSS_INVOKING_CLI=baseten
so success text names commands the user can actually run. Unset or any other
value keeps the existing truss wording.
"""

from __future__ import annotations

import os
from typing import Literal

INVOKING_CLI_ENV = "BASETEN_TRUSS_INVOKING_CLI"
InvokingCLI = Literal["truss", "baseten"]


def invoking_cli() -> InvokingCLI:
    raw = os.environ.get(INVOKING_CLI_ENV, "").strip().lower()
    if raw == "baseten":
        return "baseten"
    return "truss"


def train_logs(job_id: str) -> str:
    if invoking_cli() == "baseten":
        return f"baseten train job logs --job-id {job_id} --tail"
    return f"truss train logs --job-id {job_id} --tail"


def train_metrics(job_id: str) -> str:
    if invoking_cli() == "baseten":
        return f"baseten train job metrics --job-id {job_id}"
    return f"truss train metrics --job-id {job_id}"


def train_view(job_id: str) -> str:
    if invoking_cli() == "baseten":
        return f"baseten train job describe --job-id {job_id}"
    return f"truss train view --job-id={job_id}"


def train_stop(job_id: str) -> str:
    if invoking_cli() == "baseten":
        return f"baseten train job stop --job-id {job_id}"
    return f"truss train stop --job-id {job_id}"


def train_cache_summarize(project_name: str) -> str:
    if invoking_cli() == "baseten":
        return f"baseten train project cache describe --project {project_name}"
    return f'truss train cache summarize "{project_name}"'


def ssh_setup() -> str:
    if invoking_cli() == "baseten":
        return "baseten ssh setup"
    return "truss ssh setup"
