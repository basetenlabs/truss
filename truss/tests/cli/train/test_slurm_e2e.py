"""
End-to-end test for SLURM harness.

Usage:
    uv run pytest truss/tests/cli/train/test_slurm_e2e.py -v -s --project <name> \
        --image <image> --docker-auth-method <method> --docker-auth-secret <secret> \
        --partition <partition>

Example:
    uv run pytest truss/tests/cli/train/test_slurm_e2e.py -v -s \
        --project slurm-e2e-test \
        --image us-east4-docker.pkg.dev/calvin-calendar/sids-test-repo/sidsimage:v3 \
        --docker-auth-method gcp_service_account \
        --docker-auth-secret gcp-service-account-json \
        --partition H200
"""

import time

import pytest
from click.testing import CliRunner


@pytest.fixture
def e2e_config(request):
    return {
        "project": request.config.getoption("--project"),
        "image": request.config.getoption("--image"),
        "docker_auth_method": request.config.getoption("--docker-auth-method"),
        "docker_auth_secret": request.config.getoption("--docker-auth-secret"),
        "partition": request.config.getoption("--partition"),
        "remote": request.config.getoption("--remote"),
    }


def _get_remote_provider(remote):
    from typing import cast

    from truss.remote.baseten.remote import BasetenRemote
    from truss.remote.remote_factory import RemoteFactory

    return cast(BasetenRemote, RemoteFactory.create(remote=remote))


def _wait_for_job_status(remote_provider, job_id, target_statuses, timeout=600):
    """Poll until job reaches one of the target statuses or times out."""
    start = time.time()
    while time.time() - start < timeout:
        jobs = remote_provider.api.search_training_jobs(job_id=job_id)
        if not jobs:
            time.sleep(10)
            continue
        status = jobs[0].get("current_status", "")
        print(f"  [{int(time.time() - start)}s] {job_id}: {status}")
        if status in target_statuses:
            return jobs[0]
        if "FAILED" in status:
            pytest.fail(
                f"Job {job_id} failed: {jobs[0].get('error_message', 'unknown')}"
            )
        time.sleep(15)
    pytest.fail(f"Job {job_id} timed out after {timeout}s")


def _wait_for_login_ready(remote_provider, job_id, timeout=600):
    """Wait for the login node to be running and reach the worker-wait phase."""
    job = _wait_for_job_status(
        remote_provider, job_id, ["TRAINING_JOB_RUNNING"], timeout=timeout
    )
    # Give setup_login.sh time to install SLURM and reach the worker wait loop
    print("  Login node running, waiting 90s for SLURM setup...")
    time.sleep(90)
    return job


class TestSlurmE2E:
    def test_login_and_worker(self, e2e_config):
        """Deploy login node, wait for ready, then deploy worker and verify."""
        from truss.cli.cli import truss_cli

        runner = CliRunner()
        remote = e2e_config["remote"]
        project = e2e_config["project"]

        # --- Step 1: Deploy login node ---
        print("\n=== Step 1: Deploying login node ===")
        login_args = [
            "train",
            "slurm",
            "login",
            "--project",
            project,
            "--partition",
            e2e_config["partition"],
            "--remote",
            remote,
        ]
        if e2e_config["image"]:
            login_args.extend(["--image", e2e_config["image"]])
        if e2e_config["docker_auth_method"]:
            login_args.extend(
                ["--docker-auth-method", e2e_config["docker_auth_method"]]
            )
        if e2e_config["docker_auth_secret"]:
            login_args.extend(
                ["--docker-auth-secret", e2e_config["docker_auth_secret"]]
            )

        result = runner.invoke(truss_cli, login_args)
        print(result.output)
        assert result.exit_code == 0, f"Login push failed: {result.output}"

        # Extract job ID from output
        login_job_id = None
        for line in result.output.splitlines():
            if "pushed:" in line.lower() or "already running:" in line.lower():
                login_job_id = line.split()[-1].strip()
                break
        assert login_job_id, f"Could not extract login job ID from: {result.output}"
        print(f"  Login job ID: {login_job_id}")

        # --- Step 2: Wait for login node to be ready ---
        print("\n=== Step 2: Waiting for login node ===")
        remote_provider = _get_remote_provider(remote)
        _wait_for_login_ready(remote_provider, login_job_id, timeout=600)
        print("  Login node ready!")

        # --- Step 3: Deploy worker ---
        print("\n=== Step 3: Deploying worker ===")
        worker_args = [
            "train",
            "slurm",
            "sbatch",
            "--wrap",
            "echo SLURM_E2E_TEST_OK && hostname && nvidia-smi -L && sleep 10 && echo SLURM_E2E_TEST_DONE",
            "--project",
            project,
            "--partition",
            e2e_config["partition"],
            "--nodes",
            "1",
            "--remote",
            remote,
        ]
        if e2e_config["image"]:
            worker_args.extend(["--image", e2e_config["image"]])
        if e2e_config["docker_auth_method"]:
            worker_args.extend(
                ["--docker-auth-method", e2e_config["docker_auth_method"]]
            )
        if e2e_config["docker_auth_secret"]:
            worker_args.extend(
                ["--docker-auth-secret", e2e_config["docker_auth_secret"]]
            )

        result = runner.invoke(truss_cli, worker_args)
        print(result.output)
        assert result.exit_code == 0, f"Worker push failed: {result.output}"

        # Extract worker job ID
        worker_job_id = None
        for line in result.output.splitlines():
            if "pushed:" in line.lower():
                worker_job_id = line.split()[-1].strip()
                break
        assert worker_job_id, f"Could not extract worker job ID from: {result.output}"
        print(f"  Worker job ID: {worker_job_id}")

        # --- Step 4: Wait for worker to start running ---
        print("\n=== Step 4: Waiting for worker to run ===")
        _wait_for_job_status(
            remote_provider, worker_job_id, ["TRAINING_JOB_RUNNING"], timeout=600
        )
        print("  Worker running!")

        # --- Step 5: Verify login node detected the worker ---
        print("\n=== Step 5: Checking login node detected worker ===")
        # Wait for the login node to generate slurm.conf and start slurmctld
        time.sleep(180)

        # Check login logs for LOGIN_READY
        from truss.remote.baseten.core import get_training_job_logs_with_pagination

        jobs = remote_provider.api.search_training_jobs(job_id=login_job_id)
        project_id = jobs[0]["training_project"]["id"]
        logs = get_training_job_logs_with_pagination(
            remote_provider.api, project_id, login_job_id
        )
        log_text = "\n".join(logs)

        assert "Discovered worker count:" in log_text, (
            f"Login node never discovered workers. Last logs:\n{log_text[-2000:]}"
        )
        print("  Login node discovered workers!")

        if "LOGIN_READY" in log_text:
            print("  LOGIN_READY confirmed!")
        else:
            print(
                "  Warning: LOGIN_READY not yet in logs, slurmctld may still be starting"
            )

        # --- Step 6: Check worker is running SLURM job ---
        print("\n=== Step 6: Checking worker SLURM job ===")
        worker_logs = get_training_job_logs_with_pagination(
            remote_provider.api, project_id, worker_job_id
        )
        worker_log_text = "\n".join(worker_logs)

        if "SBATCH_RESULT:" in worker_log_text:
            print("  sbatch submitted successfully!")
        else:
            print("  Warning: sbatch not yet submitted, worker may still be setting up")

        print("\n=== E2E test passed! ===")
        print(f"  Login:  {login_job_id}")
        print(f"  Worker: {worker_job_id}")
