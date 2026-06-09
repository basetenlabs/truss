from unittest import mock

import pytest

from truss.remote import remote_factory as rf_module
from truss.remote.remote_factory import KEYRING_SERVICE, RemoteFactory
from truss.remote.truss_remote import RemoteConfig, RemoteUser, TrussRemote

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

SAMPLE_TRUSSRC_WITH_TEAM = """
[test_team]
remote_provider=test_remote
api_key=test_key
remote_url=http://test.com
team=my-team-name
"""

SAMPLE_TRUSSRC_WITH_TEAM_SPACES = """
[test_team_spaces]
remote_provider=test_remote
api_key=test_key
remote_url=http://test.com
team=my team with spaces
"""

SAMPLE_TRUSSRC_EXTRA_PARAM = """
[test_extra]
remote_provider=test_remote
api_key=test_key
remote_url=http://test.com
extra_field=some_value
"""


class TrussTestRemote(TrussRemote):
    def __init__(self, api_key, remote_url):
        self.api_key = api_key

    def authenticate(self):
        return {"Authorization": self.api_key}

    def push(self):
        return {"status": "success"}

    def get_service(self, **kwargs):
        raise NotImplementedError

    def sync_truss_to_dev_version_by_name(self, model_name: str, target_directory: str):
        raise NotImplementedError

    def whoami(self) -> RemoteUser:
        return RemoteUser("test_user", "test_email")


def mock_service_config():
    return RemoteConfig(
        name="mock-service", configs={"remote_provider": "test_remote", **SAMPLE_CONFIG}
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
@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC_EXTRA_PARAM
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_create_with_extra_param(mock_exists, mock_open):
    remote = RemoteFactory.create("test_extra")
    assert isinstance(remote, TrussTestRemote)
    assert remote.api_key == "test_key"


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC_NO_REMOTE
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_create_no_remote_param(mock_exists, mock_open):
    with pytest.raises(ValueError, match="remote_provider"):
        RemoteFactory.create("test")


@mock.patch.dict(RemoteFactory.REGISTRY, {"test_remote": TrussTestRemote}, clear=True)
@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC_NO_PARAMS
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_create_missing_required_param(mock_exists, mock_open):
    with pytest.raises(ValueError, match="Missing required parameter"):
        RemoteFactory.create("test")


@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC_WITH_TEAM
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_get_remote_team_returns_team_when_configured(mock_exists, mock_open):
    team = RemoteFactory.get_remote_team("test_team")
    assert team == "my-team-name"


@mock.patch("builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_get_remote_team_returns_none_when_not_configured(mock_exists, mock_open):
    team = RemoteFactory.get_remote_team("test")
    assert team is None


@mock.patch("pathlib.Path.exists", return_value=False)
def test_get_remote_team_returns_none_when_file_not_found(mock_exists):
    team = RemoteFactory.get_remote_team("nonexistent")
    assert team is None


@mock.patch("builtins.open", new_callable=mock.mock_open, read_data=SAMPLE_TRUSSRC)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_get_remote_team_returns_none_when_remote_not_found(mock_exists, mock_open):
    team = RemoteFactory.get_remote_team("nonexistent_remote")
    assert team is None


@mock.patch(
    "builtins.open",
    new_callable=mock.mock_open,
    read_data=SAMPLE_TRUSSRC_WITH_TEAM_SPACES,
)
@mock.patch("pathlib.Path.exists", return_value=True)
def test_get_remote_team_returns_team_with_spaces(mock_exists, mock_open):
    team = RemoteFactory.get_remote_team("test_team_spaces")
    assert team == "my team with spaces"


def test_legacy_plaintext_round_trip(trussrc):
    RemoteFactory.update_remote_config(
        RemoteConfig(
            name="legacy",
            configs={
                "remote_provider": "baseten",
                "remote_url": "http://x",
                "api_key": "plain",
            },
        )
    )
    assert "api_key = plain" in trussrc.read_text()
    loaded = RemoteFactory.load_remote_config("legacy")
    assert loaded.configs["api_key"] == "plain"
    assert "auth_type" not in loaded.configs


def test_keyring_offload_round_trip(memory_keyring, trussrc):
    RemoteFactory.update_remote_config(
        RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "auth_type": "api_key",
                "api_key": "secret",
                "remote_url": "http://x",
            },
        )
    )
    text = trussrc.read_text()
    assert "api_key = " not in text
    assert "auth_type = api_key" in text
    assert memory_keyring.get_password(KEYRING_SERVICE, "baseten") is not None

    loaded = RemoteFactory.load_remote_config("baseten")
    assert loaded.configs["api_key"] == "secret"
    assert loaded.configs["auth_type"] == "api_key"


def test_env_disabled_keeps_inline_silently(
    memory_keyring, trussrc, monkeypatch, caplog
):
    monkeypatch.setenv(rf_module.KEYRING_DISABLED_ENV, "1")
    with caplog.at_level("WARNING", logger=rf_module.logger.name):
        RemoteFactory.update_remote_config(
            RemoteConfig(
                name="baseten",
                configs={
                    "remote_provider": "baseten",
                    "auth_type": "api_key",
                    "api_key": "secret",
                    "remote_url": "http://x",
                },
            )
        )
    assert caplog.records == []
    assert "api_key = secret" in trussrc.read_text()
    assert memory_keyring.get_password(KEYRING_SERVICE, "baseten") is None

    loaded = RemoteFactory.load_remote_config("baseten")
    assert loaded.configs["api_key"] == "secret"


def test_unusable_backend_warns_and_keeps_inline(fail_keyring, trussrc, caplog):
    with caplog.at_level("WARNING", logger=rf_module.logger.name):
        RemoteFactory.update_remote_config(
            RemoteConfig(
                name="baseten",
                configs={
                    "remote_provider": "baseten",
                    "auth_type": "api_key",
                    "api_key": "secret",
                    "remote_url": "http://x",
                },
            )
        )
    assert any("plaintext" in r.message for r in caplog.records)
    assert "api_key = secret" in trussrc.read_text()

    loaded = RemoteFactory.load_remote_config("baseten")
    assert loaded.configs["api_key"] == "secret"


def test_load_raises_when_secret_missing_and_keyring_unavailable(fail_keyring, trussrc):
    trussrc.write_text(
        "[baseten]\n"
        "remote_provider = baseten\n"
        "auth_type = api_key\n"
        "remote_url = http://x\n"
    )
    with pytest.raises(ValueError, match="keyring is unavailable"):
        RemoteFactory.load_remote_config("baseten")


def test_load_raises_when_keyring_entry_missing(memory_keyring, trussrc):
    trussrc.write_text(
        "[baseten]\n"
        "remote_provider = baseten\n"
        "auth_type = api_key\n"
        "remote_url = http://x\n"
    )
    with pytest.raises(ValueError, match="No credentials in keyring"):
        RemoteFactory.load_remote_config("baseten")


def test_load_raises_when_keyring_entry_malformed(memory_keyring, trussrc):
    trussrc.write_text(
        "[baseten]\n"
        "remote_provider = baseten\n"
        "auth_type = api_key\n"
        "remote_url = http://x\n"
    )
    memory_keyring.set_password(KEYRING_SERVICE, "baseten", "not-json")
    with pytest.raises(ValueError, match="not valid JSON"):
        RemoteFactory.load_remote_config("baseten")


def test_remove_remote_config_drops_section_and_keyring(memory_keyring, trussrc):
    RemoteFactory.update_remote_config(
        RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "auth_type": "api_key",
                "api_key": "secret",
                "remote_url": "http://x",
            },
        )
    )
    assert memory_keyring.get_password(KEYRING_SERVICE, "baseten") is not None

    RemoteFactory.remove_remote_config("baseten")

    assert "[baseten]" not in trussrc.read_text()
    assert memory_keyring.get_password(KEYRING_SERVICE, "baseten") is None


def test_remove_remote_config_missing_is_noop(memory_keyring, trussrc):
    trussrc.write_text("[other]\nremote_provider = baseten\n")
    RemoteFactory.remove_remote_config("baseten")
    assert "[other]" in trussrc.read_text()
