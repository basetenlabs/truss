import json

import aiofiles
import pytest

from truss.templates.shared import dynamic_config_resolver
from truss_chains import private_types


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
def test_get_dynamic_chainlet_config_value_sync(
    config, tmp_path, dynamic_config_mount_dir
):
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(config))
    chainlet_service_config = dynamic_config_resolver.get_dynamic_config_value_sync(
        private_types.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    assert json.loads(chainlet_service_config) == config


@pytest.mark.parametrize(
    "config", [{"environment_name": "production", "foo": "bar"}, {}, "", None]
)
def test_get_dynamic_config_environment_value_sync(
    config, tmp_path, dynamic_config_mount_dir
):
    with (tmp_path / dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY).open(
        "w"
    ) as f:
        f.write(json.dumps(config))
    environment_str = dynamic_config_resolver.get_dynamic_config_value_sync(
        dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY
    )
    assert json.loads(environment_str) == config


def test_get_missing_config_value_sync(dynamic_config_mount_dir):
    chainlet_service_config = dynamic_config_resolver.get_dynamic_config_value_sync(
        private_types.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    assert not chainlet_service_config


@pytest.mark.asyncio
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
async def test_get_dynamic_chainlet_config_value_async(
    config, tmp_path, dynamic_config_mount_dir
):
    async with aiofiles.open(
        tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY, "w"
    ) as f:
        await f.write(json.dumps(config))
    chainlet_service_config = (
        await dynamic_config_resolver.get_dynamic_config_value_async(
            private_types.DYNAMIC_CHAINLET_CONFIG_KEY
        )
    )
    assert json.loads(chainlet_service_config) == config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config", [{"environment_name": "production", "foo": "bar"}, {}, "", None]
)
async def test_get_dynamic_config_environment_value_async(
    config, tmp_path, dynamic_config_mount_dir
):
    async with aiofiles.open(
        tmp_path / dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY, "w"
    ) as f:
        await f.write(json.dumps(config))
    environment_str = await dynamic_config_resolver.get_dynamic_config_value_async(
        dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY
    )
    assert json.loads(environment_str) == config


@pytest.mark.asyncio
async def test_get_missing_config_value_async(dynamic_config_mount_dir):
    chainlet_service_config = (
        await dynamic_config_resolver.get_dynamic_config_value_async(
            private_types.DYNAMIC_CHAINLET_CONFIG_KEY
        )
    )
    assert not chainlet_service_config
