import json

from truss.templates.shared.dynamic_config_resolver import get_dynamic_config_value

from truss_chains import definitions

CHAINLET_URL_MAP_VALUE = {
    "RandInt": "https://model-id.api.baseten.co/deployment/deployment-id/predict"
}


def test_get_dynamic_config_value(tmp_path, dynamic_config_mount_dir):
    with (tmp_path / definitions.DYNAMIC_CONFIG_CHAINLET_URL_MAP_KEY).open("w") as f:
        f.write(json.dumps(CHAINLET_URL_MAP_VALUE))
    chainlet_service_config = get_dynamic_config_value(
        definitions.DYNAMIC_CONFIG_CHAINLET_URL_MAP_KEY
    )
    assert json.loads(chainlet_service_config) == CHAINLET_URL_MAP_VALUE


def test_get_missing_config_value(dynamic_config_mount_dir):
    chainlet_service_config = get_dynamic_config_value(
        definitions.DYNAMIC_CONFIG_CHAINLET_URL_MAP_KEY
    )
    assert not chainlet_service_config
