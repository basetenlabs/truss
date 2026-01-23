import datetime
import sys
from pathlib import Path
from unittest import mock

import pytest

from truss.cli.utils import self_upgrade
from truss.util import user_config


class TestDetectInstallationMethod:
    @pytest.mark.parametrize(
        "prefix,pyvenv_cfg_content,conda_prefix,expected",
        [
            # conda installation (uses pip since truss isn't on conda channels)
            (
                "/home/user/miniconda3/envs/myenv",
                None,
                "/home/user/miniconda3/envs/myenv",
                ("conda", "pip install --upgrade truss", "=={version}"),
            ),
            # pip venv (pyvenv.cfg without uv)
            (
                "/home/user/project/.venv",
                "home = /usr/bin\nimplementation = CPython",
                None,
                (
                    "pip",
                    f"{sys.executable} -m pip install --upgrade truss",
                    "=={version}",
                ),
            ),
        ],
        ids=["conda", "pip_venv"],
    )
    def test_detection_scenarios(
        self, prefix, pyvenv_cfg_content, conda_prefix, expected, tmp_path, monkeypatch
    ):
        fake_prefix = tmp_path / "fake_prefix"
        fake_prefix.mkdir()
        actual_prefix = str(fake_prefix)

        monkeypatch.setattr(sys, "prefix", actual_prefix)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        if conda_prefix:
            monkeypatch.setenv("CONDA_PREFIX", actual_prefix)
        else:
            monkeypatch.delenv("CONDA_PREFIX", raising=False)

        if pyvenv_cfg_content:
            pyvenv_cfg = Path(actual_prefix) / "pyvenv.cfg"
            pyvenv_cfg.write_text(pyvenv_cfg_content)

        result = self_upgrade.detect_installation_method()

        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert result.method == expected[0]
            assert result.version_suffix == expected[2]
            assert expected[1] in result.upgrade_command

    @pytest.mark.parametrize(
        "installer,dist_path,expected_method,expected_cmd,expected_version_fmt",
        [
            (
                "uv",
                "/home/user/.local/share/uv/tools/truss",
                "uv",
                "uv tool install --force truss",
                "@{version}",
            ),
            (
                "uv",
                "/home/user/project/.venv/lib/python3.11/site-packages",
                "uv",
                "uv pip install --upgrade truss",
                "=={version}",
            ),
            (
                "pipx",
                "/home/user/.local/pipx/venvs/truss",
                "pipx",
                "pipx upgrade truss",
                "=={version}",
            ),
            (
                "pip",
                "/home/user/.local/share/pipx/venvs/truss",
                "pipx",
                "pipx upgrade truss",
                "=={version}",
            ),
        ],
        ids=["uv_tool", "uv_venv", "pipx_installer", "pipx_path"],
    )
    def test_detection_via_installer_metadata(
        self,
        installer,
        dist_path,
        expected_method,
        expected_cmd,
        expected_version_fmt,
        tmp_path,
        monkeypatch,
    ):
        fake_prefix = tmp_path / "fake_prefix"
        fake_prefix.mkdir()
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(
            self_upgrade, "_get_installer_info", lambda: (installer, Path(dist_path))
        )

        result = self_upgrade.detect_installation_method()

        assert result is not None
        assert result.method == expected_method
        assert expected_cmd in result.upgrade_command
        assert result.version_suffix == expected_version_fmt

    def test_pip_installer_falls_through_to_pyvenv_check(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "venv"
        fake_prefix.mkdir()
        (fake_prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(
            self_upgrade,
            "_get_installer_info",
            lambda: (
                "pip",
                Path("/home/user/project/.venv/lib/python3.11/site-packages"),
            ),
        )

        result = self_upgrade.detect_installation_method()

        assert result is not None
        assert result.method == "pip"
        assert "pip install --upgrade truss" in result.upgrade_command

    def test_detection_fails_when_no_indicators(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "unknown_env"
        fake_prefix.mkdir()
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        result = self_upgrade.detect_installation_method()
        assert result is None


class TestRunUpgrade:
    def test_run_upgrade_success(self, tmp_path, monkeypatch, capsys):
        fake_prefix = tmp_path / "venv"
        fake_prefix.mkdir()
        (fake_prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with mock.patch.object(self_upgrade.console, "input", return_value="y"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                self_upgrade.run_upgrade()

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "pip install --upgrade truss" in call_args[0][0]

    def test_run_upgrade_with_version_pip(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "venv"
        fake_prefix.mkdir()
        (fake_prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with mock.patch.object(self_upgrade.console, "input", return_value="y"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                self_upgrade.run_upgrade("0.12.3")

        call_args = mock_run.call_args
        assert "truss==0.12.3" in call_args[0][0]

    def test_run_upgrade_with_version_conda(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "conda_env"
        fake_prefix.mkdir()
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.setenv("CONDA_PREFIX", str(fake_prefix))
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with mock.patch.object(self_upgrade.console, "input", return_value="y"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                self_upgrade.run_upgrade("0.12.3")

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "pip install --upgrade truss==0.12.3" in cmd

    def test_run_upgrade_cancelled(self, tmp_path, monkeypatch, capsys):
        fake_prefix = tmp_path / "venv"
        fake_prefix.mkdir()
        (fake_prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with mock.patch.object(self_upgrade.console, "input", return_value="n"):
            with mock.patch("subprocess.run") as mock_run:
                self_upgrade.run_upgrade()

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()

    def test_run_upgrade_non_interactive(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "venv"
        fake_prefix.mkdir()
        (fake_prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with mock.patch.object(self_upgrade.console, "input") as mock_input:
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                self_upgrade.run_upgrade(interactive=False)

        mock_input.assert_not_called()
        mock_run.assert_called_once()

    def test_run_upgrade_detection_fails(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "unknown"
        fake_prefix.mkdir()
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with pytest.raises(SystemExit) as exc_info:
            self_upgrade.run_upgrade()

        assert exc_info.value.code == 1

    def test_run_upgrade_subprocess_fails(self, tmp_path, monkeypatch):
        fake_prefix = tmp_path / "venv"
        fake_prefix.mkdir()
        (fake_prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(self_upgrade, "_get_installer_info", lambda: None)

        with mock.patch.object(self_upgrade.console, "input", return_value="y"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=1)
                with pytest.raises(SystemExit) as exc_info:
                    self_upgrade.run_upgrade()

        assert exc_info.value.code == 1


class TestNotifyIfOutdated:
    def test_notifies_when_outdated(self, monkeypatch, capsys):
        mock_update_info = user_config.UpdateInfo(
            upgrade_recommended=True, reason="outdated", latest_version="0.12.3"
        )
        mock_state = mock.Mock()
        mock_state.should_notify_upgrade.return_value = mock_update_info
        mock_settings = mock.Mock()
        mock_settings.check_for_updates = True

        monkeypatch.setattr(user_config, "state", mock_state)
        monkeypatch.setattr(user_config, "settings", mock_settings)

        self_upgrade.notify_if_outdated("0.11.0")

        captured = capsys.readouterr()
        assert "0.12.3" in captured.out
        assert "0.11.0" in captured.out
        assert "truss upgrade" in captured.out
        assert "check_for_updates" in captured.out

    def test_no_notification_when_up_to_date(self, monkeypatch, capsys):
        mock_state = mock.Mock()
        mock_state.should_notify_upgrade.return_value = None
        mock_settings = mock.Mock()
        mock_settings.check_for_updates = True

        monkeypatch.setattr(user_config, "state", mock_state)
        monkeypatch.setattr(user_config, "settings", mock_settings)

        self_upgrade.notify_if_outdated("0.12.3")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_skips_check_when_check_for_updates_is_false(self, monkeypatch, capsys):
        mock_state = mock.Mock()
        mock_settings = mock.Mock()
        mock_settings.check_for_updates = False

        monkeypatch.setattr(user_config, "state", mock_state)
        monkeypatch.setattr(user_config, "settings", mock_settings)

        self_upgrade.notify_if_outdated("0.11.0")

        captured = capsys.readouterr()
        assert captured.out == ""
        mock_state.should_notify_upgrade.assert_not_called()

    def test_raises_on_exception(self, monkeypatch):
        mock_state = mock.Mock()
        mock_state.should_notify_upgrade.side_effect = Exception("Network error")
        mock_settings = mock.Mock()
        mock_settings.check_for_updates = True

        monkeypatch.setattr(user_config, "state", mock_state)
        monkeypatch.setattr(user_config, "settings", mock_settings)

        # Exception handling moved to upgrade_dialogue() in common.py
        with pytest.raises(Exception, match="Network error"):
            self_upgrade.notify_if_outdated("0.11.0")


class TestShouldNotifyUpgrade:
    def test_returns_update_info_when_outdated(self, tmp_path):
        state = user_config.State(
            version_info=user_config.VersionInfo(
                latest_version="0.12.3", last_check=datetime.datetime.min
            )
        )
        wrapper = user_config._StateWrapper(state)

        with mock.patch.object(wrapper, "should_upgrade") as mock_should_upgrade:
            mock_should_upgrade.return_value = user_config.UpdateInfo(
                upgrade_recommended=True, reason="outdated", latest_version="0.12.3"
            )
            result = wrapper.should_notify_upgrade("0.11.0")

        assert result is not None
        assert result.latest_version == "0.12.3"

    def test_returns_none_when_upgrade_not_recommended(self, tmp_path):
        state = user_config.State()
        wrapper = user_config._StateWrapper(state)

        with mock.patch.object(wrapper, "should_upgrade") as mock_should_upgrade:
            mock_should_upgrade.return_value = user_config.UpdateInfo(
                upgrade_recommended=False, reason="up to date", latest_version="0.12.3"
            )
            result = wrapper.should_notify_upgrade("0.12.3")

        assert result is None
