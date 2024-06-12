import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, TypeVar

import psutil
import requests

BLOB_DOWNLOAD_TIMEOUT_SECS = 600  # 10 minutes
# number of seconds to wait for truss server child processes before sending kill signal
CHILD_PROCESS_WAIT_TIMEOUT_SECONDS = 120


def model_supports_predict_proba(model: object) -> bool:
    if not hasattr(model, "predict_proba"):
        return False
    if hasattr(model, "_check_proba"):  # noqa eg Support Vector Machines *can* predict proba if they made certain choices while training
        try:
            model._check_proba()
            return True
        except AttributeError:
            return False
    return True


def cpu_count():
    """Get the available CPU count for this system.
    Takes the minimum value from the following locations:
    - Total system cpus available on the host.
    - CPU Affinity (if set)
    - Cgroups limit (if set)
    """
    count = os.cpu_count()

    # Check CPU affinity if available
    try:
        affinity_count = len(psutil.Process().cpu_affinity())
        if affinity_count > 0:
            count = min(count, affinity_count)
    except Exception:
        pass

    # Check cgroups if available
    if sys.platform == "linux":
        try:
            with open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us") as f:
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us") as f:
                period = int(f.read())
            cgroups_count = int(quota / period)
            if cgroups_count > 0:
                count = min(count, cgroups_count)
        except Exception:
            pass

    return count


def all_processes_dead(procs: List[multiprocessing.Process]) -> bool:
    for proc in procs:
        if proc.is_alive():
            return False
    return True


def kill_child_processes(parent_pid: int):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.terminate()
    gone, alive = psutil.wait_procs(
        children, timeout=CHILD_PROCESS_WAIT_TIMEOUT_SECONDS
    )
    for process in alive:
        process.kill()


X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


def transform_keys(d: Dict[X, Z], fn: Callable[[X], Y]) -> Dict[Y, Z]:
    return {fn(key): value for key, value in d.items()}


def download_from_url_using_requests(URL: str, download_to: Path):
    # Streaming download to keep memory usage low
    resp = requests.get(
        URL,
        allow_redirects=True,
        stream=True,
        timeout=BLOB_DOWNLOAD_TIMEOUT_SECS,
    )
    resp.raise_for_status()
    with download_to.open("wb") as file:
        shutil.copyfileobj(resp.raw, file)
