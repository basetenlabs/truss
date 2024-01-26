import os
import sys

import rich
from truss.cli.cli import _get_truss_from_directory
from truss.remote.baseten.core import ModelName
from truss.remote.remote_factory import RemoteFactory

remote = "prod"
target_directory = os.getcwd()

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
