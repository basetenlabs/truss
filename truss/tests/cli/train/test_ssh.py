from unittest import mock

import pytest


class TestParseHostname:
    """Tests for proxy_command.parse_hostname."""

    def _parse(self, hostname):
        from truss.cli.train.proxy_command import parse_hostname

        return parse_hostname(hostname)

    def test_basic(self):
        assert self._parse("training-job-5wo5n3y-0.dev.ssh.baseten.co") == (
            "5wo5n3y",
            "0",
            "dev",
            None,
        )

    def test_multi_digit_replica(self):
        assert self._parse("training-job-abc1234-12.baseten.ssh.baseten.co") == (
            "abc1234",
            "12",
            "baseten",
            None,
        )

    def test_remote_with_dashes(self):
        assert self._parse(
            "training-job-5wo5n3y-0.my-custom-remote.ssh.baseten.co"
        ) == ("5wo5n3y", "0", "my-custom-remote", None)

    def test_staging_remote(self):
        assert self._parse("training-job-abc1234-2.staging.ssh.baseten.co") == (
            "abc1234",
            "2",
            "staging",
            None,
        )

    def test_api_prefix(self):
        assert self._parse("training-job-rwn61qy-0.dev.mc-dev.ssh.baseten.co") == (
            "rwn61qy",
            "0",
            "dev",
            "mc-dev",
        )

    def test_invalid_no_suffix(self):
        with pytest.raises(SystemExit):
            self._parse("training-job-5wo5n3y-0.dev.example.com")

    def test_invalid_no_prefix(self):
        with pytest.raises(SystemExit):
            self._parse("5wo5n3y-0.dev.ssh.baseten.co")  # missing training-job- prefix

    def test_no_remote(self):
        assert self._parse("training-job-5wo5n3y-0.ssh.baseten.co") == (
            "5wo5n3y",
            "0",
            None,
            None,
        )

    def test_invalid_no_replica(self):
        with pytest.raises(SystemExit):
            self._parse("training-job-5wo5n3y.dev.ssh.baseten.co")  # no -node


class TestLoadTrussrc:
    """Tests for proxy_command.load_trussrc."""

    def test_loads_remote(self, tmp_path):
        from truss.cli.train.proxy_command import load_trussrc

        rc = tmp_path / ".trussrc"
        rc.write_text(
            "[dev]\n"
            "remote_provider = baseten\n"
            "api_key = test-key-123\n"
            "remote_url = https://app.dev.baseten.co\n"
        )
        with mock.patch("truss.cli.train.proxy_command.TRUSSRC_PATH", rc):
            api_key, remote_url = load_trussrc("dev")
        assert api_key == "test-key-123"
        assert remote_url == "https://app.dev.baseten.co"

    def test_missing_remote(self, tmp_path):
        from truss.cli.train.proxy_command import load_trussrc

        rc = tmp_path / ".trussrc"
        rc.write_text("[baseten]\napi_key = x\nremote_url = y\n")
        with mock.patch("truss.cli.train.proxy_command.TRUSSRC_PATH", rc):
            with pytest.raises(SystemExit):
                load_trussrc("nonexistent")

    def test_missing_file(self, tmp_path):
        from truss.cli.train.proxy_command import load_trussrc

        with mock.patch(
            "truss.cli.train.proxy_command.TRUSSRC_PATH", tmp_path / "nope"
        ):
            with pytest.raises(SystemExit):
                load_trussrc("dev")


class TestResolveRestApiUrl:
    def test_default_is_prod(self):
        from truss.cli.train.proxy_command import resolve_rest_api_url

        assert resolve_rest_api_url() == "https://api.baseten.co"

    def test_api_prefix(self):
        from truss.cli.train.proxy_command import resolve_rest_api_url

        assert resolve_rest_api_url("mc-dev") == "https://api.mc-dev.baseten.co"

    def test_staging_prefix(self):
        from truss.cli.train.proxy_command import resolve_rest_api_url

        assert resolve_rest_api_url("staging") == "https://api.staging.baseten.co"


class TestResolveRemote:
    def test_explicit_remote_passthrough(self):
        import configparser

        from truss.cli.train.proxy_command import resolve_remote

        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n")
        assert resolve_remote("dev", config) == "dev"

    def test_single_remote_default(self):
        import configparser

        from truss.cli.train.proxy_command import resolve_remote

        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n")
        assert resolve_remote(None, config) == "dev"

    def test_multiple_remotes_errors(self):
        import configparser

        from truss.cli.train.proxy_command import resolve_remote

        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n\n[baseten]\napi_key = y\n")
        with pytest.raises(SystemExit):
            resolve_remote(None, config)


class TestEnsureSSHKeypair:
    def test_creates_keypair_ed25519(self, tmp_path):
        from truss.cli.train.ssh import ensure_ssh_keypair

        key_dir = tmp_path / "ssh" / "baseten"
        mock_result = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
            result = ensure_ssh_keypair(key_dir)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "ssh-keygen"
            assert "ed25519" in args
            assert result == key_dir / "id_ed25519"

    def test_falls_back_to_rsa(self, tmp_path):
        from truss.cli.train.ssh import ensure_ssh_keypair

        key_dir = tmp_path / "ssh" / "baseten"
        ed25519_fail = mock.Mock(returncode=1)
        rsa_ok = mock.Mock(returncode=0)
        with mock.patch(
            "subprocess.run", side_effect=[ed25519_fail, rsa_ok]
        ) as mock_run:
            result = ensure_ssh_keypair(key_dir)
            assert mock_run.call_count == 2
            rsa_args = mock_run.call_args_list[1][0][0]
            assert "rsa" in rsa_args
            assert result == key_dir / "id_rsa"

    def test_skips_existing_ed25519(self, tmp_path):
        from truss.cli.train.ssh import ensure_ssh_keypair

        key_dir = tmp_path / "ssh" / "baseten"
        key_dir.mkdir(parents=True)
        (key_dir / "id_ed25519").touch()

        with mock.patch("subprocess.run") as mock_run:
            result = ensure_ssh_keypair(key_dir)
            mock_run.assert_not_called()
            assert result == key_dir / "id_ed25519"

    def test_skips_existing_rsa(self, tmp_path):
        from truss.cli.train.ssh import ensure_ssh_keypair

        key_dir = tmp_path / "ssh" / "baseten"
        key_dir.mkdir(parents=True)
        (key_dir / "id_rsa").touch()

        with mock.patch("subprocess.run") as mock_run:
            result = ensure_ssh_keypair(key_dir)
            mock_run.assert_not_called()
            assert result == key_dir / "id_rsa"


class TestInstallProxyCommandScript:
    def test_installs_with_version(self, tmp_path):
        from truss.cli.train.ssh import install_proxy_command_script

        key_dir = tmp_path / "ssh" / "baseten"
        with mock.patch("truss.__version__", "1.2.3"):
            dest = install_proxy_command_script(key_dir)

        assert dest.exists()
        content = dest.read_text()
        assert 'CLIENT_VERSION = "1.2.3"' in content
        assert "{{CLIENT_VERSION}}" not in content


class TestSetupSSHConfig:
    def test_creates_new_config(self, tmp_path):
        from truss.cli.train.ssh import MARKER_END, MARKER_START, setup_ssh_config

        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        with mock.patch("truss.cli.train.ssh.SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert MARKER_START in content
        assert MARKER_END in content
        assert "*.ssh.baseten.co" in content
        assert "proxy-command.py" in content
        assert "User baseten" in content

    def test_replaces_existing_block(self, tmp_path):
        from truss.cli.train.ssh import MARKER_END, MARKER_START, setup_ssh_config

        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        ssh_config.write_text(
            "Host other-server\n    User admin\n\n"
            f"{MARKER_START}\nOLD BLOCK\n{MARKER_END}\n\n"
            "Host another-server\n    User root\n"
        )

        with mock.patch("truss.cli.train.ssh.SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert content.count(MARKER_START) == 1
        assert "OLD BLOCK" not in content
        assert "Host other-server" in content
        assert "Host another-server" in content
        assert "*.ssh.baseten.co" in content

    def test_preserves_other_entries(self, tmp_path):
        from truss.cli.train.ssh import setup_ssh_config

        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        existing = "Host myserver\n    User deploy\n    Port 2222\n"
        ssh_config.write_text(existing)

        with mock.patch("truss.cli.train.ssh.SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert "Host myserver" in content
        assert "User deploy" in content
        assert "Port 2222" in content
