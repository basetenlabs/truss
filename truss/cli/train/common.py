from typing import Optional, cast

import rich_click as click

from truss.cli.utils.output import console
from truss.remote.baseten import BasetenRemote


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
