import json
from contextlib import contextmanager
from pathlib import Path

from truss.templates.shared.extra_config_resolver import ExtraConfigResolver

CHAINLET_SERVICE_CONFIG_NAME = "chainlet_service_config"
CHAINLET_SERVICE_CONFIG_VALUE = {
    "RandInt": {
        "name": "RandInt",
        "predict_url": "https://model-id.api.baseten.co/deployment/deployment-id/predict",
    }
}


def test_get_config_value(tmp_path):
    with (tmp_path / CHAINLET_SERVICE_CONFIG_NAME).open("w") as f:
        f.write(json.dumps(CHAINLET_SERVICE_CONFIG_VALUE))
    with _extra_config_mount_dir(tmp_path):
        chainlet_service_config = ExtraConfigResolver.get_config_value(
            CHAINLET_SERVICE_CONFIG_NAME
        )
        assert json.loads(chainlet_service_config) == CHAINLET_SERVICE_CONFIG_VALUE


def test_get_missing_config_value(tmp_path):
    with _extra_config_mount_dir(tmp_path):
        chainlet_service_config = ExtraConfigResolver.get_config_value(
            CHAINLET_SERVICE_CONFIG_NAME
        )
        assert not chainlet_service_config


@contextmanager
def _extra_config_mount_dir(path: Path):
    orig_extra_config_mount_dir = ExtraConfigResolver.EXTRA_CONFIG_MOUNT_DIR
    ExtraConfigResolver.EXTRA_CONFIG_MOUNT_DIR = str(path)
    try:
        yield
    finally:
        ExtraConfigResolver.EXTRA_CONFIG_MOUNT_DIR = orig_extra_config_mount_dir
