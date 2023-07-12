from unittest import mock

import pytest
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import TrussRemote

SAMPLE_CONFIG = {"api_key": "test_key", "remote_url": "http://test.com"}


class TestRemote(TrussRemote):
    def __init__(self, api_key, remote_url):
        self.api_key = api_key
        self.remote_url = remote_url

    def authenticate(self):
        return {"Authorization": self.api_key}

    def push(self):
        return {"status": "success"}


def mock_service_config():
    return {"remote": "test_remote", **SAMPLE_CONFIG}


def mock_incorrect_service_config():
    return {"remote": "nonexistent_remote", **SAMPLE_CONFIG}


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
@mock.patch(
    "truss.remote.remote_factory.RemoteFactory.load_service",
    return_value=mock_service_config(),
)
def test_create(mock_load_service):
    service_name = "test_service"
    remote = RemoteFactory.create(service_name)
    mock_load_service.assert_called_once_with(service_name)
    assert isinstance(remote, TestRemote)


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
@mock.patch(
    "truss.remote.remote_factory.RemoteFactory.load_service",
    return_value=(mock_incorrect_service_config()),
)
def test_create_no_service(mock_load_service):
    service_name = "nonexistent_service"
    with pytest.raises(ValueError):
        RemoteFactory.create(service_name)


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
def test_load_service():
    sample_trussrc = """
    [test]
    remote=test_remote
    api_key=test_key
    remote_url=http://test.com
    """
    with mock.patch("builtins.open", mock.mock_open(read_data=sample_trussrc)):
        service = RemoteFactory.load_service("test")
        assert service == {"remote": "test_remote", **SAMPLE_CONFIG}


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
def test_load_service_no_file():
    with mock.patch("builtins.open", mock.MagicMock(side_effect=FileNotFoundError())):
        with pytest.raises(ValueError):
            RemoteFactory.load_service("test")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
def test_load_service_no_service():
    sample_trussrc = """
    [different]
    remote=test_remote
    api_key=test_key
    """
    with mock.patch("builtins.open", mock.mock_open(read_data=sample_trussrc)):
        with pytest.raises(ValueError):
            RemoteFactory.load_service("test")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
def test_required_params():
    required_params = RemoteFactory.required_params(TestRemote)
    assert required_params == {"api_key", "remote_url"}


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
def test_validate_service_no_remote():
    sample_trussrc = """
    [test]
    api_key=test_key
    """
    with mock.patch("builtins.open", mock.mock_open(read_data=sample_trussrc)):
        with pytest.raises(ValueError):
            service = RemoteFactory.load_service("test")
            RemoteFactory.validate_service(service, "test")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TestRemote}, clear=True)
def test_load_service_no_params():
    sample_trussrc = """
    [test]
    remote=test_remote
    """
    with mock.patch("builtins.open", mock.mock_open(read_data=sample_trussrc)):
        with pytest.raises(ValueError):
            service = RemoteFactory.load_service("test")
            RemoteFactory.validate_service(service, "test")
