import json

import pytest
from truss.templates.shared.dynamic_config_resolver import get_dynamic_config_value

from truss_chains import definitions


@pytest.mark.parametrize(
    "config",
    [
        {
            "RandInt": {
                "predict_url": "https://model-id.api.baseten.co/deployment/deployment-id/predict"
            }
        },
        {},
        "",
    ],
)
def test_get_dynamic_config_value(config, tmp_path, dynamic_config_mount_dir):
    with (tmp_path / definitions.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(config))
    chainlet_service_config = get_dynamic_config_value(
        definitions.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    assert json.loads(chainlet_service_config) == config


def test_get_missing_config_value(dynamic_config_mount_dir):
    chainlet_service_config = get_dynamic_config_value(
        definitions.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    assert not chainlet_service_config
