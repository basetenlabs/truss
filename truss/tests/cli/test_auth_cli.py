from unittest.mock import patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote import remote_factory as rf_module
from truss.remote.baseten.oauth import OAuthCredential, OAuthError
from truss.remote.remote_factory import KEYRING_SERVICE, RemoteFactory


def test_login_browser_writes_oauth_config(memory_keyring, trussrc):
    cred = OAuthCredential(
        access_token="atk", refresh_token="rtk", expires_at=9999999999
    )
    with patch("truss.cli.auth.oauth.run_device_flow", return_value=cred) as mock_flow:
        result = CliRunner().invoke(truss_cli, ["auth", "login", "--browser"])

    assert result.exit_code == 0, result.output
    mock_flow.assert_called_once_with("https://api.baseten.co")
    loaded = RemoteFactory.load_remote_config("baseten")
    assert loaded.configs["auth_type"] == "oauth"
    assert loaded.configs["oauth_access_token"] == "atk"


def test_login_browser_propagates_oauth_error(memory_keyring, trussrc):
    with patch(
        "truss.cli.auth.oauth.run_device_flow", side_effect=OAuthError("denied")
    ):
        result = CliRunner().invoke(truss_cli, ["auth", "login", "--browser"])
    assert result.exit_code != 0
    assert "denied" in result.output


def test_login_browser_and_api_key_mutually_exclusive(trussrc):
    result = CliRunner().invoke(
        truss_cli, ["auth", "login", "--browser", "--api-key", "k"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_login_api_key_writes_config(memory_keyring, trussrc):
    result = CliRunner().invoke(truss_cli, ["auth", "login", "--api-key", "abc"])
    assert result.exit_code == 0, result.output
    loaded = RemoteFactory.load_remote_config("baseten")
    assert loaded.configs["auth_type"] == "api_key"
    assert loaded.configs["api_key"] == "abc"


def test_login_no_flags_non_interactive_errors(trussrc):
    with patch("truss.cli.auth.check_is_interactive", return_value=False):
        result = CliRunner().invoke(truss_cli, ["auth", "login"])
    assert result.exit_code == 2
    assert "--browser" in result.output
    assert "--api-key" in result.output


def test_truss_login_alias_runs_browser_flow(memory_keyring, trussrc):
    cred = OAuthCredential(access_token="a", refresh_token="r", expires_at=9999999999)
    with patch("truss.cli.auth.oauth.run_device_flow", return_value=cred):
        result = CliRunner().invoke(truss_cli, ["login", "--browser"])
    assert result.exit_code == 0, result.output
    assert RemoteFactory.load_remote_config("baseten").configs["auth_type"] == "oauth"


def test_logout_oauth_revokes_and_removes(memory_keyring, trussrc):
    RemoteFactory.update_remote_config(
        rf_module.RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "remote_url": "http://x",
                "auth_type": "oauth",
                "oauth_access_token": "atk",
                "oauth_refresh_token": "rtk",
                "oauth_expires_at": "9999999999",
            },
        )
    )
    with patch("truss.cli.auth.oauth.revoke") as mock_revoke:
        result = CliRunner().invoke(truss_cli, ["auth", "logout"])
    assert result.exit_code == 0, result.output
    mock_revoke.assert_called_once()
    assert "[baseten]" not in trussrc.read_text()
    assert memory_keyring.get_password(KEYRING_SERVICE, "baseten") is None


def test_logout_api_key_skips_revoke(memory_keyring, trussrc):
    RemoteFactory.update_remote_config(
        rf_module.RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "remote_url": "http://x",
                "auth_type": "api_key",
                "api_key": "k",
            },
        )
    )
    with patch("truss.cli.auth.oauth.revoke") as mock_revoke:
        result = CliRunner().invoke(truss_cli, ["auth", "logout"])
    assert result.exit_code == 0, result.output
    mock_revoke.assert_not_called()
    assert "[baseten]" not in trussrc.read_text()


def test_logout_unknown_remote_errors(trussrc):
    trussrc.write_text("")
    result = CliRunner().invoke(truss_cli, ["auth", "logout", "--remote", "missing"])
    assert result.exit_code != 0
    assert "Not logged in" in result.output


def test_status_keyring_source(memory_keyring, trussrc):
    RemoteFactory.update_remote_config(
        rf_module.RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "remote_url": "http://x",
                "auth_type": "api_key",
                "api_key": "secret",
            },
        )
    )
    result = CliRunner().invoke(truss_cli, ["auth", "status"])
    assert result.exit_code == 0, result.output
    assert "auth_type: api_key" in result.output
    assert "source: keyring" in result.output


def test_status_legacy_plaintext(trussrc):
    trussrc.write_text(
        "[baseten]\nremote_provider = baseten\nremote_url = http://x\napi_key = legacy\n"
    )
    result = CliRunner().invoke(truss_cli, ["auth", "status"])
    assert result.exit_code == 0, result.output
    assert "legacy plaintext" in result.output
    assert "source: trussrc-inline" in result.output


def test_status_missing_remote_errors(trussrc):
    trussrc.write_text("")
    result = CliRunner().invoke(truss_cli, ["auth", "status", "--remote", "missing"])
    assert result.exit_code != 0


def test_login_browser_writes_named_remote(memory_keyring, trussrc):
    cred = OAuthCredential(access_token="a", refresh_token="r", expires_at=9999999999)
    with patch("truss.cli.auth.oauth.run_device_flow", return_value=cred):
        result = CliRunner().invoke(
            truss_cli, ["auth", "login", "--browser", "--remote", "staging"]
        )
    assert result.exit_code == 0, result.output
    loaded = RemoteFactory.load_remote_config("staging")
    assert loaded.configs["auth_type"] == "oauth"
    assert loaded.configs["remote_url"] == "https://app.baseten.co"


def test_login_api_key_writes_named_remote(memory_keyring, trussrc):
    result = CliRunner().invoke(
        truss_cli, ["auth", "login", "--api-key", "k", "--remote", "alt"]
    )
    assert result.exit_code == 0, result.output
    loaded = RemoteFactory.load_remote_config("alt")
    assert loaded.configs["auth_type"] == "api_key"
    assert loaded.configs["api_key"] == "k"


def test_login_reuses_existing_remote_url(memory_keyring, trussrc):
    trussrc.write_text(
        "[staging]\nremote_provider = baseten\n"
        "remote_url = https://app.staging.baseten.co\n"
    )
    cred = OAuthCredential(access_token="a", refresh_token="r", expires_at=9999999999)
    with patch("truss.cli.auth.oauth.run_device_flow", return_value=cred) as mock_flow:
        result = CliRunner().invoke(
            truss_cli, ["auth", "login", "--browser", "--remote", "staging"]
        )
    assert result.exit_code == 0, result.output
    mock_flow.assert_called_once_with("https://api.staging.baseten.co")
    loaded = RemoteFactory.load_remote_config("staging")
    assert loaded.configs["remote_url"] == "https://app.staging.baseten.co"


def test_login_prints_success_message(memory_keyring, trussrc):
    result = CliRunner().invoke(
        truss_cli, ["auth", "login", "--api-key", "k", "--remote", "alt"]
    )
    assert result.exit_code == 0, result.output
    assert "Logged in to remote" in result.output
    assert "alt" in result.output
