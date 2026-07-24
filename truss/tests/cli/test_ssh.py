import configparser
import ssl
import sys
import urllib.error
from unittest import mock

import pytest

from truss.cli import proxy_command
from truss.cli import ssh as ssh_mod
from truss.cli.proxy_command import (
    WORKLOAD_MODEL,
    WORKLOAD_TRAINING,
    ParsedHostname,
    load_trussrc,
    parse_hostname,
    resolve_remote,
    resolve_rest_api_url,
)
from truss.cli.ssh import (
    MARKER_END,
    MARKER_START,
    SSH_CONFIG_BLOCK_WINDOWS,
    ensure_ssh_keypair,
    install_proxy_command_script,
    is_setup_complete,
    setup_ssh_config,
)


class TestParseHostname:
    """Tests for proxy_command.parse_hostname."""

    def _parse(self, hostname):
        return parse_hostname(hostname)

    def _training(self, id, replica, remote, api_prefix):
        return ParsedHostname(
            workload_type=WORKLOAD_TRAINING,
            id=id,
            replica=replica,
            deployment_id=None,
            remote=remote,
            api_prefix=api_prefix,
        )

    def _model(self, id, deployment_id, replica, remote, api_prefix):
        return ParsedHostname(
            workload_type=WORKLOAD_MODEL,
            id=id,
            replica=replica,
            deployment_id=deployment_id,
            remote=remote,
            api_prefix=api_prefix,
        )

    def test_basic(self):
        assert self._parse(
            "training-job-5wo5n3y-0.dev.ssh.baseten.co"
        ) == self._training("5wo5n3y", "0", "dev", None)

    def test_multi_digit_replica(self):
        assert self._parse(
            "training-job-abc1234-12.baseten.ssh.baseten.co"
        ) == self._training("abc1234", "12", "baseten", None)

    def test_remote_with_dashes(self):
        assert self._parse(
            "training-job-5wo5n3y-0.my-custom-remote.ssh.baseten.co"
        ) == self._training("5wo5n3y", "0", "my-custom-remote", None)

    def test_staging_remote(self):
        assert self._parse(
            "training-job-abc1234-2.staging.ssh.baseten.co"
        ) == self._training("abc1234", "2", "staging", None)

    def test_api_prefix(self):
        assert self._parse(
            "training-job-rwn61qy-0.dev.mc-dev.ssh.baseten.co"
        ) == self._training("rwn61qy", "0", "dev", "mc-dev")

    def test_invalid_no_suffix(self):
        with pytest.raises(SystemExit):
            self._parse("training-job-5wo5n3y-0.dev.example.com")

    def test_invalid_no_prefix(self):
        with pytest.raises(SystemExit):
            self._parse("5wo5n3y-0.dev.ssh.baseten.co")

    def test_no_remote(self):
        assert self._parse("training-job-5wo5n3y-0.ssh.baseten.co") == self._training(
            "5wo5n3y", "0", None, None
        )

    def test_invalid_no_replica(self):
        with pytest.raises(SystemExit):
            self._parse("training-job-5wo5n3y.dev.ssh.baseten.co")

    def test_model_with_deployment(self):
        assert self._parse("model-abc123-def456.dev.ssh.baseten.co") == self._model(
            "abc123", "def456", None, "dev", None
        )

    def test_model_with_deployment_and_replica(self):
        assert self._parse(
            "model-abc123-def456-ghi789.dev.ssh.baseten.co"
        ) == self._model("abc123", "def456", "ghi789", "dev", None)

    def test_model_replica_with_dashes(self):
        assert self._parse(
            "model-abc123-def456-pod-xyz-abc.dev.ssh.baseten.co"
        ) == self._model("abc123", "def456", "pod-xyz-abc", "dev", None)

    def test_model_no_remote(self):
        assert self._parse("model-abc123-def456.ssh.baseten.co") == self._model(
            "abc123", "def456", None, None, None
        )

    def test_model_api_prefix(self):
        assert self._parse(
            "model-abc123-def456.dev.mc-dev.ssh.baseten.co"
        ) == self._model("abc123", "def456", None, "dev", "mc-dev")

    def test_model_missing_deployment(self):
        with pytest.raises(SystemExit):
            self._parse("model-abc123.dev.ssh.baseten.co")

    def test_model_empty_id(self):
        with pytest.raises(SystemExit):
            self._parse("model-.dev.ssh.baseten.co")

    def test_model_empty_deployment(self):
        with pytest.raises(SystemExit):
            self._parse("model-abc123-.dev.ssh.baseten.co")

    def test_model_empty_replica(self):
        with pytest.raises(SystemExit):
            self._parse("model-abc123-def456-.dev.ssh.baseten.co")


class TestLoadTrussrc:
    """Tests for proxy_command.load_trussrc."""

    def test_loads_remote(self, tmp_path):
        rc = tmp_path / ".trussrc"
        rc.write_text(
            "[dev]\n"
            "remote_provider = baseten\n"
            "api_key = test-key-123\n"
            "remote_url = https://app.dev.baseten.co\n"
        )
        with mock.patch("truss.cli.proxy_command.TRUSSRC_PATH", rc):
            api_key, remote_url = load_trussrc("dev")
        assert api_key == "test-key-123"
        assert remote_url == "https://app.dev.baseten.co"

    def test_missing_remote(self, tmp_path):
        rc = tmp_path / ".trussrc"
        rc.write_text("[baseten]\napi_key = x\nremote_url = y\n")
        with mock.patch("truss.cli.proxy_command.TRUSSRC_PATH", rc):
            with pytest.raises(SystemExit):
                load_trussrc("nonexistent")

    def test_missing_file(self, tmp_path):
        with mock.patch("truss.cli.proxy_command.TRUSSRC_PATH", tmp_path / "nope"):
            with pytest.raises(SystemExit):
                load_trussrc("dev")


class TestResolveRestApiUrl:
    def test_default_is_prod(self):
        assert resolve_rest_api_url() == "https://api.baseten.co"

    def test_api_prefix(self):
        assert resolve_rest_api_url("mc-dev") == "https://api.mc-dev.baseten.co"

    def test_staging_prefix(self):
        assert resolve_rest_api_url("staging") == "https://api.staging.baseten.co"


class TestResolveRemote:
    def test_explicit_remote_passthrough(self):
        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n")
        assert resolve_remote("dev", config) == "dev"

    def test_single_remote_default(self):
        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n")
        assert resolve_remote(None, config) == "dev"

    def test_multiple_remotes_errors(self):
        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n\n[baseten]\napi_key = y\n")
        with mock.patch.object(proxy_command, "DEFAULT_REMOTE", ""):
            with pytest.raises(SystemExit):
                resolve_remote(None, config)

    def test_multiple_remotes_with_default(self):
        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n\n[baseten]\napi_key = y\n")
        with mock.patch.object(proxy_command, "DEFAULT_REMOTE", "baseten"):
            assert resolve_remote(None, config) == "baseten"

    def test_explicit_remote_overrides_default(self):
        config = configparser.ConfigParser()
        config.read_string("[dev]\napi_key = x\n\n[baseten]\napi_key = y\n")
        with mock.patch.object(proxy_command, "DEFAULT_REMOTE", "baseten"):
            assert resolve_remote("dev", config) == "dev"


class TestEnsureSSHKeypair:
    def test_creates_keypair_ed25519(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        mock_result = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
            key_path, reused = ensure_ssh_keypair(key_dir)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "ssh-keygen"
            assert "ed25519" in args
            assert key_path == key_dir / "id_ed25519"
            assert reused is False

    def test_falls_back_to_rsa(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        ed25519_fail = mock.Mock(returncode=1)
        rsa_ok = mock.Mock(returncode=0)
        with mock.patch(
            "subprocess.run", side_effect=[ed25519_fail, rsa_ok]
        ) as mock_run:
            key_path, reused = ensure_ssh_keypair(key_dir)
            assert mock_run.call_count == 2
            rsa_args = mock_run.call_args_list[1][0][0]
            assert "rsa" in rsa_args
            assert key_path == key_dir / "id_rsa"
            assert reused is False

    def test_skips_existing_ed25519(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        key_dir.mkdir(parents=True)
        (key_dir / "id_ed25519").touch()

        with mock.patch("subprocess.run") as mock_run:
            key_path, reused = ensure_ssh_keypair(key_dir)
            mock_run.assert_not_called()
            assert key_path == key_dir / "id_ed25519"
            assert reused is True

    def test_skips_existing_rsa(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        key_dir.mkdir(parents=True)
        (key_dir / "id_rsa").touch()

        with mock.patch("subprocess.run") as mock_run:
            key_path, reused = ensure_ssh_keypair(key_dir)
            mock_run.assert_not_called()
            assert key_path == key_dir / "id_rsa"
            assert reused is True


class TestInstallProxyCommandScript:
    def test_installs_with_version(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        with mock.patch("truss.__version__", "1.2.3"):
            dest = install_proxy_command_script(key_dir)

        assert dest.exists()
        content = dest.read_text()
        assert 'CLIENT_VERSION = "1.2.3"' in content
        assert "{{CLIENT_VERSION}}" not in content

    def test_installs_with_default_remote(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        with mock.patch("truss.__version__", "1.2.3"):
            dest = install_proxy_command_script(key_dir, default_remote="baseten")

        content = dest.read_text()
        assert 'DEFAULT_REMOTE = "baseten"' in content
        assert "{{DEFAULT_REMOTE}}" not in content

    def test_installs_without_default_remote(self, tmp_path):
        key_dir = tmp_path / "ssh" / "baseten"
        with mock.patch("truss.__version__", "1.2.3"):
            dest = install_proxy_command_script(key_dir)

        content = dest.read_text()
        assert 'DEFAULT_REMOTE = ""' in content
        assert "{{DEFAULT_REMOTE}}" not in content


class TestSetupSSHConfig:
    @pytest.fixture(autouse=True)
    def _stub_resolve_python(self):
        with mock.patch.object(
            ssh_mod, "_resolve_python", return_value="/usr/bin/python3"
        ):
            yield

    def test_creates_new_config(self, tmp_path):
        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        with mock.patch.object(ssh_mod, "SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert MARKER_START in content
        assert MARKER_END in content
        assert "training-job-*.ssh.baseten.co" in content
        assert "model-*.ssh.baseten.co" in content
        assert "proxy-command.py" in content
        assert "User baseten" in content
        assert "User app" in content

    def test_replaces_existing_block(self, tmp_path):
        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        ssh_config.write_text(
            "Host other-server\n    User admin\n\n"
            f"{MARKER_START}\nOLD BLOCK\n{MARKER_END}\n\n"
            "Host another-server\n    User root\n"
        )

        with mock.patch.object(ssh_mod, "SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert content.count(MARKER_START) == 1
        assert "OLD BLOCK" not in content
        assert "Host other-server" in content
        assert "Host another-server" in content
        assert "*.ssh.baseten.co" in content

    def test_preserves_other_entries(self, tmp_path):
        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        existing = "Host myserver\n    User deploy\n    Port 2222\n"
        ssh_config.write_text(existing)

        with mock.patch.object(ssh_mod, "SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert "Host myserver" in content
        assert "User deploy" in content
        assert "Port 2222" in content

    def test_replace_preserves_first_char_of_next_entry(self, tmp_path):
        """Regression: slice must not consume a char if MARKER_END has no trailing newline."""
        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        ssh_config.write_text(
            f"{MARKER_START}\nOLD BLOCK\n{MARKER_END}Host another-server\n    User root\n"
        )

        with mock.patch.object(ssh_mod, "SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert "Host another-server" in content  # full "Host" preserved, not "ost"
        assert "User root" in content

    def test_replace_handles_crlf_line_endings(self, tmp_path):
        """Regression: slice should consume \\r\\n as a single separator, not half of it."""
        ssh_config = tmp_path / "config"
        key_dir = tmp_path / "baseten"
        key_dir.mkdir()

        ssh_config.write_bytes(
            f"{MARKER_START}\r\nOLD BLOCK\r\n{MARKER_END}\r\nHost another-server\r\n    User root\r\n".encode()
        )

        with mock.patch.object(ssh_mod, "SSH_CONFIG_PATH", ssh_config):
            setup_ssh_config(key_dir)

        content = ssh_config.read_text()
        assert "Host another-server" in content
        assert "OLD BLOCK" not in content


class TestWindowsConfigBlock:
    """Regression tests for Windows-specific quoting in SSH config block."""

    def _render(self):
        return SSH_CONFIG_BLOCK_WINDOWS.format(
            marker_start=MARKER_START,
            marker_end=MARKER_END,
            python=r"C:\Program Files\Python311\python.exe",
            proxy_script=r"C:\Users\bob\.ssh\baseten\proxy-command.py",
            key_path=r"C:\Users\bob\.ssh\baseten\id_ed25519",
            cert_path=r"C:\Users\bob\.ssh\baseten\id_ed25519-cert.pub",
        )

    def test_match_exec_escapes_quotes_around_paths(self):
        rendered = self._render()
        assert (
            r'exec "\"C:\Program Files\Python311\python.exe\" '
            r'\"C:\Users\bob\.ssh\baseten\proxy-command.py\" --sign %n"' in rendered
        )

    def test_proxycommand_quotes_paths(self):
        rendered = self._render()
        assert (
            r'ProxyCommand "C:\Program Files\Python311\python.exe" '
            r'"C:\Users\bob\.ssh\baseten\proxy-command.py" %n' in rendered
        )

    def test_no_broken_cmd_c_wrapper(self):
        rendered = self._render()
        assert "cmd /c" not in rendered
        assert "if exist" not in rendered


class TestIsSetupComplete:
    def test_returns_true_with_ed25519(self, tmp_path):
        (tmp_path / "proxy-command.py").touch()
        (tmp_path / "id_ed25519").touch()
        assert is_setup_complete(tmp_path) is True

    def test_returns_true_with_rsa(self, tmp_path):
        (tmp_path / "proxy-command.py").touch()
        (tmp_path / "id_rsa").touch()
        assert is_setup_complete(tmp_path) is True

    def test_returns_false_without_proxy_script(self, tmp_path):
        (tmp_path / "id_ed25519").touch()
        assert is_setup_complete(tmp_path) is False

    def test_returns_false_without_key(self, tmp_path):
        (tmp_path / "proxy-command.py").touch()
        assert is_setup_complete(tmp_path) is False


class TestProbePython:
    def test_probes_current_interpreter(self):
        probed = ssh_mod._probe_python(sys.executable)
        assert probed is not None
        version, has_certs = probed
        assert version == (sys.version_info.major, sys.version_info.minor)
        assert isinstance(has_certs, bool)

    def test_unusable_interpreter_returns_none(self):
        assert ssh_mod._probe_python("/nonexistent/python") is None


class TestResolvePython:
    def _mock_which(self, mapping):
        return mock.patch("shutil.which", side_effect=lambda name: mapping.get(name))

    def _mock_probe(self, probes):
        return mock.patch.object(
            ssh_mod, "_probe_python", side_effect=lambda path: probes.get(path)
        )

    def test_prefers_interpreter_with_certs(self):
        which = {
            "python3.13": "/usr/local/bin/python3.13",
            "python3.12": "/opt/homebrew/bin/python3.12",
        }
        probes = {
            "/usr/local/bin/python3.13": ((3, 13), False),
            "/opt/homebrew/bin/python3.12": ((3, 12), True),
        }
        with self._mock_which(which), self._mock_probe(probes):
            assert ssh_mod._resolve_python() == "/opt/homebrew/bin/python3.12"

    def test_all_missing_certs_raises_cert_error(self):
        which = {"python3": "/usr/local/bin/python3"}
        probes = {"/usr/local/bin/python3": ((3, 12), False)}
        with self._mock_which(which), self._mock_probe(probes):
            with pytest.raises(RuntimeError, match="SSL root certificates"):
                ssh_mod._resolve_python()

    def test_cert_error_names_checked_interpreters(self):
        which = {"python3": "/usr/local/bin/python3"}
        probes = {"/usr/local/bin/python3": ((3, 12), False)}
        with self._mock_which(which), self._mock_probe(probes):
            with pytest.raises(RuntimeError, match="/usr/local/bin/python3"):
                ssh_mod._resolve_python()

    def test_no_suitable_version_raises_version_error(self):
        which = {"python3": "/usr/bin/python3"}
        probes = {"/usr/bin/python3": ((3, 9), True)}
        with self._mock_which(which), self._mock_probe(probes):
            with pytest.raises(RuntimeError, match="Could not find Python 3.10"):
                ssh_mod._resolve_python()

    def test_old_version_without_certs_raises_version_error(self):
        which = {"python3": "/usr/bin/python3"}
        probes = {"/usr/bin/python3": ((3, 9), False)}
        with self._mock_which(which), self._mock_probe(probes):
            with pytest.raises(RuntimeError, match="Could not find Python 3.10"):
                ssh_mod._resolve_python()

    def test_nothing_found_raises_version_error(self):
        with self._mock_which({}), self._mock_probe({}):
            with pytest.raises(RuntimeError, match="Could not find Python 3.10"):
                ssh_mod._resolve_python()

    def test_skips_venv_python(self):
        which = {
            "python3.12": "/repo/.venv/bin/python3.12",
            "python3": "/usr/bin/python3",
        }
        probes = {"/usr/bin/python3": ((3, 12), True)}
        with self._mock_which(which), self._mock_probe(probes) as probe:
            assert ssh_mod._resolve_python() == "/usr/bin/python3"
            probe.assert_called_once_with("/usr/bin/python3")


class TestPythonMissingRootCerts:
    def test_true_for_empty_store(self):
        with mock.patch.object(ssh_mod, "_probe_python", return_value=((3, 12), False)):
            assert ssh_mod.python_missing_root_certs("/some/python") is True

    def test_false_with_certs(self):
        with mock.patch.object(ssh_mod, "_probe_python", return_value=((3, 12), True)):
            assert ssh_mod.python_missing_root_certs("/some/python") is False

    def test_false_for_unusable_interpreter(self):
        with mock.patch.object(ssh_mod, "_probe_python", return_value=None):
            assert ssh_mod.python_missing_root_certs("/some/python") is False


class TestTrustStoreEmpty:
    def _mock_ssl(self, ca_count, cafile=None, capath=None):
        fake_ctx = mock.Mock()
        fake_ctx.cert_store_stats.return_value = {"x509_ca": ca_count}
        fake_paths = mock.Mock(cafile=cafile, capath=capath)
        return (
            mock.patch.object(
                proxy_command.ssl, "create_default_context", return_value=fake_ctx
            ),
            mock.patch.object(
                proxy_command.ssl, "get_default_verify_paths", return_value=fake_paths
            ),
        )

    def test_populated_store(self):
        ctx_patch, paths_patch = self._mock_ssl(140)
        with ctx_patch, paths_patch:
            assert proxy_command._trust_store_empty() is False

    def test_empty_store_no_paths(self):
        ctx_patch, paths_patch = self._mock_ssl(0)
        with ctx_patch, paths_patch:
            assert proxy_command._trust_store_empty() is True

    def test_empty_store_with_cafile(self, tmp_path):
        cafile = tmp_path / "cert.pem"
        cafile.write_text("PEM")
        ctx_patch, paths_patch = self._mock_ssl(0, cafile=str(cafile))
        with ctx_patch, paths_patch:
            assert proxy_command._trust_store_empty() is False

    def test_empty_store_with_lazy_capath(self, tmp_path):
        (tmp_path / "abcd1234.0").write_text("PEM")
        ctx_patch, paths_patch = self._mock_ssl(0, capath=str(tmp_path))
        with ctx_patch, paths_patch:
            assert proxy_command._trust_store_empty() is False

    def test_empty_store_with_empty_capath(self, tmp_path):
        ctx_patch, paths_patch = self._mock_ssl(0, capath=str(tmp_path))
        with ctx_patch, paths_patch:
            assert proxy_command._trust_store_empty() is True


class TestTlsCertError:
    def test_empty_store_message_names_interpreter(self, capsys):
        with mock.patch.object(proxy_command, "_trust_store_empty", return_value=True):
            with pytest.raises(SystemExit):
                proxy_command.tls_cert_error(Exception("boom"))
        err = capsys.readouterr().err
        assert "no SSL root certificates" in err
        assert sys.executable in err

    def test_populated_store_gets_generic_message(self, capsys):
        with mock.patch.object(proxy_command, "_trust_store_empty", return_value=False):
            with pytest.raises(SystemExit):
                proxy_command.tls_cert_error(Exception("boom"))
        err = capsys.readouterr().err
        assert "TLS certificate verification failed: boom" in err

    def test_api_request_routes_cert_failures(self, capsys):
        exc = urllib.error.URLError(
            ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")
        )
        with mock.patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(SystemExit):
                proxy_command.api_request("https://api.baseten.co/v1/x", "key")
        assert "TLS certificate verification failed" in capsys.readouterr().err
