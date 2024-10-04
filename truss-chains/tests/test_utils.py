import json

import pytest

from truss_chains import definitions
from truss_chains.utils import override_chainlet_to_service_metadata

CHAINLET_URL_MAP_NAME = "chainlet_url_map"
CHAINLET_URL_MAP_VALUE = {
    "HelloWorld": "https://model-diff_id.api.baseten.co/deployment/diff_deployment_id/predict"
}


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def test_override_chainlet_to_service_metadata(tmp_path, dynamic_config_mount_dir):
    with (tmp_path / CHAINLET_URL_MAP_NAME).open("w") as f:
        f.write(json.dumps(CHAINLET_URL_MAP_VALUE))

    chainlet_to_service = {
        "HelloWorld": definitions.ServiceDescriptor(
            name="HelloWorld",
            predict_url="https://model-model_id.api.baseten.co/deployments/deployment_id/predict",
            options=definitions.RPCOptions(),
        )
    }
    new_chainlet_to_service = override_chainlet_to_service_metadata(chainlet_to_service)

    assert (
        new_chainlet_to_service["HelloWorld"].predict_url
        == CHAINLET_URL_MAP_VALUE["HelloWorld"]
    )


def test_no_override_chainlet_to_service_metadata(tmp_path, dynamic_config_mount_dir):
    with (tmp_path / CHAINLET_URL_MAP_NAME).open("w") as f:
        f.write(json.dumps(CHAINLET_URL_MAP_VALUE))

    chainlet_to_service = {
        "RandInt": definitions.ServiceDescriptor(
            name="HelloWorld",
            predict_url="https://model-model_id.api.baseten.co/deployments/deployment_id/predict",
            options=definitions.RPCOptions(),
        )
    }
    new_chainlet_to_service = override_chainlet_to_service_metadata(chainlet_to_service)

    assert new_chainlet_to_service == chainlet_to_service


def test_no_config_override_chainlet_to_service_metadata(
    tmp_path, dynamic_config_mount_dir
):
    chainlet_to_service = {
        "HelloWorld": definitions.ServiceDescriptor(
            name="HelloWorld",
            predict_url="https://model-model_id.api.baseten.co/deployments/deployment_id/predict",
            options=definitions.RPCOptions(),
        )
    }
    new_chainlet_to_service = override_chainlet_to_service_metadata(chainlet_to_service)

    assert new_chainlet_to_service == chainlet_to_service
