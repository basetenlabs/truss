from unittest import mock

import pytest
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import RemoteConfig, TrussRemote

SAMPLE_CONFIG = {"api_key": "test_key", "remote_url": "http://test.com"}

SAMPLE_TRUSSRC = """
[test]
remote_provider=test_remote
api_key=test_key
remote_url=http://test.com
"""

SAMPLE_TRUSSRC_NO_REMOTE = """
[test]
api_key=test_key
remote_url=http://test.com
"""

SAMPLE_TRUSSRC_NO_PARAMS = """
[test]
remote_provider=test_remote
"""


class TrussTestRemote(TrussRemote):
    def __init__(self, api_key, remote_url):
        self.api_key = api_key
        self.remote_url = remote_url

    def authenticate(self):
        return {"Authorization": self.api_key}

    def push(self):
        return {"status": "success"}

    def get_service(self, **kwargs):
        raise NotImplementedError

    def sync_truss_to_dev_version_by_name(self, model_name: str, target_directory: str):
        raise NotImplementedError


def mock_service_config():
    return RemoteConfig(
        name="mock-service",
        configs={"remote_provider": "test_remote", **SAMPLE_CONFIG},
    )


def mock_incorrect_service_config():
    return RemoteConfig(
        name="mock-incorrect-service",
        configs={"remote_provider": "nonexistent_remote", **SAMPLE_CONFIG},
    )


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch(
    "truss.remote.remote_factory.RemoteFactory.load_remote_config",
    return_value=mock_service_config(),
)
def test_create(mock_load_remote_config):
    service_name = "test_service"
    remote = RemoteFactory.create(service_name)
    mock_load_remote_config.assert_called_once_with(service_name)
    assert isinstance(remote, TrussTestRemote)


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch(
    "truss.remote.remote_factory.RemoteFactory.load_remote_config",
    return_value=mock_incorrect_service_config(),
)
def test_create_no_service(mock_load_remote_config):
    service_name = "nonexistent_service"
    with pytest.raises(ValueError):
        RemoteFactory.create(service_name)


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_load_remote_config(mock_exists, mock_open):
    service = RemoteFactory.load_remote_config("test")
    assert service.name == "test"
    assert service.configs == {"remote_provider": "test_remote", **SAMPLE_CONFIG}


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC)
@mock.patch("pathlib.Path.exists", return_value=False)
def test_load_remote_config_no_file(mock_exists, mock_open):
    with pytest.raises(FileNotFoundError):
        RemoteFactory.load_remote_config("test")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_load_remote_config_no_service(mock_exists, mock_open):
    with pytest.raises(ValueError):
        RemoteFactory.load_remote_config("nonexistent_service")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
def test_required_params():
    required_params = RemoteFactory.required_params(TrussTestRemote)
    assert required_params == {"api_key", "remote_url"}


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC_NO_REMOTE
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_validate_remote_config_no_remote(mock_exists, mock_open):
    service = RemoteFactory.load_remote_config("test")
    with pytest.raises(ValueError):
        RemoteFactory.validate_remote_config(service.configs, "test")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC_NO_PARAMS
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_load_remote_config_no_params(mock_exists, mock_open):
    service = RemoteFactory.load_remote_config("test")
    with pytest.raises(ValueError):
        RemoteFactory.validate_remote_config(service.configs, "test")
