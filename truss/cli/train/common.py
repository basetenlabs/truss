import datetime
import re
from typing import Optional, cast

import rich_click as click

from truss.cli.utils.output import console
from truss.remote.baseten import BasetenRemote

# Byte size constants
BYTES_PER_KB = 1000
BYTES_PER_MB = BYTES_PER_KB * 1000
BYTES_PER_GB = BYTES_PER_MB * 1000
BYTES_PER_TB = BYTES_PER_GB * 1000


def get_most_recent_job(
    remote_provider: BasetenRemote, project_id: Optional[str], job_id: Optional[str]
) -> tuple[str, str]:
    if not project_id or not job_id:
        jobs = remote_provider.api.search_training_jobs(
            project_id=project_id, job_id=job_id
        )
        if not jobs:
            raise click.UsageError("No jobs found.")
        if len(jobs) > 1:
            sorted_jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)
            job = sorted_jobs[0]
            console.print(
                f"Multiple jobs found. Showing the most recently created job: {job['id']}",
                style="yellow",
            )
        else:
            job = jobs[0]
        project_id = cast(str, job["training_project"]["id"])
        job_id = cast(str, job["id"])
    return project_id, job_id


def format_bytes_to_human_readable(bytes: int) -> str:
    if bytes > BYTES_PER_TB:
        return f"{bytes / BYTES_PER_TB:.2f} TB"
    if bytes > BYTES_PER_GB:
        return f"{bytes / BYTES_PER_GB:.2f} GB"
    elif bytes > BYTES_PER_MB:
        return f"{bytes / BYTES_PER_MB:.2f} MB"
    elif bytes > BYTES_PER_KB:
        return f"{bytes / BYTES_PER_KB:.2f} KB"
    else:
        return f"{bytes} B"


def _normalize_iso_timestamp(iso_timestamp: str) -> str:
    iso_timestamp = iso_timestamp.strip()
    if iso_timestamp.endswith("Z"):
        iso_timestamp = iso_timestamp[:-1] + "+00:00"

    tz_part = ""
    tz_match = re.search(r"([+-]\d{2}:\d{2}|[+-]\d{4})$", iso_timestamp)
    if tz_match:
        tz_part = tz_match.group(0)
        iso_timestamp = iso_timestamp[: tz_match.start()]

    iso_timestamp = iso_timestamp.rstrip()

    if tz_part and ":" not in tz_part:
        tz_part = f"{tz_part[:3]}:{tz_part[3:]}"

    fractional_match = re.search(r"\.(\d+)$", iso_timestamp)
    if fractional_match:
        fractional_digits = fractional_match.group(1)
        if len(fractional_digits) > 6:
            iso_timestamp = (
                iso_timestamp[: fractional_match.start()] + "." + fractional_digits[:6]
            )

    return f"{iso_timestamp}{tz_part}"


# NOTE: `pyproject.toml` declares support down to Python 3.9, whose
# `datetime.fromisoformat` cannot parse nanosecond fractions or colonless offsets,
# so normalize timestamps before parsing.
def format_localized_time(iso_timestamp: str) -> str:
    try:
        utc_time = datetime.datetime.fromisoformat(iso_timestamp)
    except ValueError:
        # Handle non-standard formats (nanoseconds, Z suffix, colonless offsets)
        normalized_timestamp = _normalize_iso_timestamp(iso_timestamp)
        utc_time = datetime.datetime.fromisoformat(normalized_timestamp)

    local_time = utc_time.astimezone()
    return local_time.strftime("%Y-%m-%d %H:%M:%S")
