# This file contains shared code to be used in other tests
# TODO(pankaj): Using a tests file for shared code is not ideal, we should
# move it to a regular file. This is a short term hack.
import json
import logging
import shutil
import subprocess
import time
from contextlib import contextmanager

from truss.base.constants import TRUSS
from truss.util.docker import get_containers, kill_all

DISK_SPACE_LOW_PERCENTAGE = 20


@contextmanager
def ensure_kill_all():
    try:
        with _show_container_logs_if_raised():
            yield
    finally:
        kill_all_with_retries()
        ensure_free_disk_space()


def _human_readable_json_logs(raw_logs: str) -> str:
    output = []
    for line in raw_logs.splitlines():
        try:
            log_entry = json.loads(line)
            human_readable_log = " ".join(
                f"{key}: {value}" for key, value in log_entry.items()
            )
            output.append(f"\t{human_readable_log}")
        except Exception:
            output.append(line)
    return "\n".join(output)


@contextmanager
def _show_container_logs_if_raised():
    initial_ids = {c.id for c in get_containers({TRUSS: True})}
    exception_raised = False
    try:
        yield
    except Exception:
        exception_raised = True
        raise
    finally:
        if exception_raised:
            print("An exception was raised, showing logs of all containers.")
            containers = get_containers({TRUSS: True})
            new_containers = [c for c in containers if c.id not in initial_ids]
            parts = ["\n"]
            for container in new_containers:
                parts.append(f"Logs for container {container.name} ({container.id}):")
                parts.append(_human_readable_json_logs(container.logs()))
                parts.append("\n")
            logging.warning("\n".join(parts))


def get_container_logs_from_prefix(prefix: str) -> str:
    containers = get_containers({TRUSS: True})
    for container in containers:
        if container.name.startswith(prefix):
            return _human_readable_json_logs(container.logs())
    return ""


def kill_all_with_retries(num_retries: int = 10):
    kill_all()
    attempts = 0
    while attempts < num_retries:
        containers = get_containers({TRUSS: True})
        if len(containers) == 0:
            return
        attempts += 1
        time.sleep(1)


def ensure_free_disk_space():
    """Check if disk space is low."""
    if is_disk_space_low():
        clear_disk_space()


def is_disk_space_low() -> bool:
    disk_usage = shutil.disk_usage("/")
    disk_free_percent = 100 * float(disk_usage.free) / disk_usage.total
    return disk_free_percent <= DISK_SPACE_LOW_PERCENTAGE


def clear_disk_space():
    docker_ps_output = subprocess.check_output(
        ["docker", "ps", "-a", "-f", "status=exited", "-q"]
    ).decode("utf-8")
    docker_containers = docker_ps_output.split("\n")[:-1]
    subprocess.run(["docker", "rm", *docker_containers])
    subprocess.run(["docker", "system", "prune", "-a", "-f"])
