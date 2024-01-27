import os
import signal
import sys

import rich
import yappi
from truss.cli.cli import _get_truss_from_directory
from truss.remote.baseten.core import ModelName
from truss.remote.remote_factory import RemoteFactory


def watch(target_directory: str, remote: str) -> None:
    remote_provider = RemoteFactory.create(remote=remote)

    tr = _get_truss_from_directory(target_directory=target_directory)
    model_name = tr.spec.config.model_name
    if not model_name:
        rich.print(
            "🧐 NoneType model_name provided in config.yaml. "
            "Please check that you have the correct model name in your config file."
        )
        sys.exit(1)

    service = remote_provider.get_service(model_identifier=ModelName(model_name))
    logs_url = remote_provider.get_remote_logs_url(service)
    rich.print(f"🪵  View logs for your deployment at {logs_url}")
    remote_provider.sync_truss_to_dev_version_by_name(model_name, target_directory)


def save_yappi_stats():
    yappi.stop()
    stats = yappi.get_func_stats()
    stats.save("profile_truss_watch.out", type="pstat")
    print("Yappi profiling stats saved to profile_truss_watch.out")


def signal_handler(sig, frame):
    save_yappi_stats()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

yappi.start(profile_threads=True)

remote = "prod"
target_directory = os.getcwd()

try:
    watch(target_directory, remote)
except KeyboardInterrupt:
    pass
