# This file contains shared code to be used in other tests
# TODO(pankaj): Using a tests file for shared code is not ideal, we should
# move it to a regular file. This is a short term hack.

import shutil
import subprocess
import time
from contextlib import contextmanager

from truss.build import kill_all
from truss.constants import TRUSS
from truss.docker import get_containers

DISK_SPACE_LOW_PERCENTAGE = 20


@contextmanager
def ensure_kill_all():
    try:
        yield
    finally:
        kill_all_with_retries()
        ensure_free_disk_space()


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
