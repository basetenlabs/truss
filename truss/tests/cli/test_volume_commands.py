import io
import json
import signal
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from truss.cli import volume_commands
from truss.cli.cannery import config as cannery_config
from truss.cli.cannery import runner as cannery_runner
from truss.cli.cli import truss_cli


class FakeProcess:
    def __init__(self, stdout='{"protocol_version":1}', stderr="", return_code=0):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = return_code
        self.signals = []
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


@pytest.fixture(autouse=True)
def clean_cannery_environment(monkeypatch, tmp_path):
    for variable in (
        "TRUSS_CANNERY_BIN",
        "TRUSS_CANNERY_API",
        "TRUSS_CANNERY_ORG",
        "TRUSS_CANNERY_AUTH_TOKEN_FILE",
        "CANNERY_AUTH_TOKEN_FILE",
        "CANNERY_CORRELATION_ID",
        "CANNERY_DIAGNOSTIC_LOG",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TRUSS_CANNERY_DIAGNOSTIC_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(
        cannery_config.RemoteFactory, "get_available_config_names", lambda: []
    )


def install_process(monkeypatch, process):
    popen = Mock(return_value=process)
    monkeypatch.setattr(
        cannery_runner, "resolve_cannery_binary", lambda **_kwargs: "/bin/cannery"
    )
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)
    return popen


def test_binary_resolution_prefers_configured_binary(monkeypatch):
    monkeypatch.setenv("TRUSS_CANNERY_BIN", "/opt/cannery")
    which = Mock(return_value="/opt/cannery")
    monkeypatch.setattr(volume_commands.shutil, "which", which)

    assert volume_commands.resolve_cannery_binary() == "/opt/cannery"
    which.assert_called_once_with("/opt/cannery")


def test_binary_resolution_falls_back_to_path(monkeypatch):
    which = Mock(return_value="/usr/local/bin/cannery")
    monkeypatch.setattr(volume_commands.shutil, "which", which)

    assert (
        volume_commands.resolve_cannery_binary(allow_path_fallback=True)
        == "/usr/local/bin/cannery"
    )
    which.assert_called_once_with("cannery")


def test_binary_resolution_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(volume_commands.shutil, "which", lambda _: None)

    with pytest.raises(click.ClickException, match="TRUSS_CANNERY_BIN"):
        volume_commands.resolve_cannery_binary()


def test_runner_builds_argv_and_environment(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("token")
    token_file.chmod(0o600)
    monkeypatch.setenv("TRUSS_CANNERY_API", "https://cannery.example.com")
    monkeypatch.setenv("TRUSS_CANNERY_ORG", "acme")
    monkeypatch.setenv("TRUSS_CANNERY_AUTH_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("CANNERY_AUTH_TOKEN_FILE", "/must/not/leak")
    popen = install_process(
        monkeypatch, FakeProcess(stdout='{"protocol_version":1,"refs":[]}')
    )

    result = volume_commands.run_cannery(["ls", "models", "--all"])
    correlation_id = result.pop("correlation_id")
    assert result == {"protocol_version": 1, "refs": []}
    assert correlation_id == popen.call_args.kwargs["env"]["CANNERY_CORRELATION_ID"]

    argv = popen.call_args.args[0]
    assert argv == [
        "/bin/cannery",
        "-o",
        "json",
        "--progress",
        "machine",
        "--api",
        "https://cannery.example.com",
        "ls",
        "models",
        "--all",
    ]
    assert popen.call_args.kwargs["env"]["CANNERY_ORG"] == "acme"
    assert popen.call_args.kwargs["env"]["CANNERY_AUTH_TOKEN_FILE"] == str(token_file)


@pytest.mark.parametrize(
    "api",
    [
        "http://localhost:8787",
        "http://127.0.0.1:8787",
        "http://127.42.0.1:8787",
        "http://[::1]:8787",
    ],
)
def test_loopback_endpoint_allows_no_token(monkeypatch, api):
    monkeypatch.setenv("TRUSS_CANNERY_API", api)
    popen = install_process(monkeypatch, FakeProcess())

    volume_commands.run_cannery(["ls"])

    assert "CANNERY_AUTH_TOKEN_FILE" not in popen.call_args.kwargs["env"]


def test_default_api_and_org_are_local_dev(monkeypatch):
    popen = install_process(monkeypatch, FakeProcess())

    volume_commands.run_cannery(["ls"])

    argv = popen.call_args.args[0]
    assert argv[argv.index("--api") + 1] == "http://127.0.0.1:8787"
    assert popen.call_args.kwargs["env"]["CANNERY_ORG"] == "dev"


def test_non_loopback_endpoint_rejected_without_token(monkeypatch):
    monkeypatch.setenv("TRUSS_CANNERY_API", "https://cannery.example.com")
    popen = Mock()
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)

    with pytest.raises(click.UsageError, match="RUN-869"):
        volume_commands.run_cannery(["ls"])

    popen.assert_not_called()


def test_missing_explicit_token_file_rejected_before_subprocess(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUSS_CANNERY_AUTH_TOKEN_FILE", str(tmp_path / "missing-token"))
    popen = Mock()
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)

    with pytest.raises(click.UsageError, match="existing token file"):
        volume_commands.run_cannery(["ls"])

    popen.assert_not_called()


def test_result_parser_requires_one_final_object(monkeypatch):
    install_process(
        monkeypatch, FakeProcess(stdout='  {"protocol_version":1,"digest":"b3:abc"}\n')
    )

    result = volume_commands.run_cannery(["show", "bdn://dev/model"])
    result.pop("correlation_id")
    assert result == {"protocol_version": 1, "digest": "b3:abc"}


@pytest.mark.parametrize("stdout", ["", "[]", "{}", '{"protocol_version":2}', "{}\n{}"])
def test_result_parser_rejects_invalid_result_contract(monkeypatch, stdout):
    install_process(monkeypatch, FakeProcess(stdout=stdout))

    with pytest.raises(volume_commands.CanneryProtocolError):
        volume_commands.run_cannery(["ls"])


def test_protocol_mismatch_fails(monkeypatch):
    event = json.dumps({"protocol_version": 2, "type": "status", "phase": "start"})
    install_process(monkeypatch, FakeProcess(stderr=f"{event}\n"))

    with pytest.raises(
        volume_commands.CanneryProtocolError, match="requires version 1"
    ):
        volume_commands.run_cannery(["ls"])


def test_ndjson_progress_is_drained_and_rendered(monkeypatch, capsys):
    events = [
        {
            "protocol_version": 1,
            "type": "status",
            "operation": "push",
            "phase": "scanning",
            "message": "bare-credential-value-9f4c",
        },
        {
            "protocol_version": 1,
            "type": "progress",
            "operation": "push",
            "phase": "upload",
            "files_done": 2,
            "files_total": 5,
            "bytes_done": 100,
            "bytes_total": 400,
            "message": "human text must not be used",
        },
    ]
    stderr = "".join(f"{json.dumps(event)}\n" for event in events)
    install_process(monkeypatch, FakeProcess(stderr=stderr))

    volume_commands.run_cannery(["push", "/tmp/model"])

    captured = capsys.readouterr()
    assert "Cannery push — scanning" in captured.err
    assert "2/5 files" in captured.err
    assert "100/400 bytes" in captured.err
    assert "human text must not be used" not in captured.err
    assert "bare-credential-value-9f4c" not in captured.err
    assert captured.out == ""


def test_invalid_ndjson_fails_loudly(monkeypatch):
    install_process(monkeypatch, FakeProcess(stderr="not-json\n"))

    with pytest.raises(volume_commands.CanneryProtocolError, match="invalid NDJSON"):
        volume_commands.run_cannery(["ls"])


def test_machine_error_and_exit_one_become_click_exception(monkeypatch):
    event = {
        "protocol_version": 1,
        "type": "error",
        "reason": "unauthorized",
        "message": "token expired",
        "hint": "create a new token",
    }
    install_process(
        monkeypatch, FakeProcess(stderr=f"{json.dumps(event)}\n", return_code=1)
    )

    with pytest.raises(click.ClickException, match="reason unauthorized") as exc:
        volume_commands.run_cannery(["ls"])

    assert "token expired" not in str(exc.value)
    assert "create a new token" not in str(exc.value)


def test_exit_two_becomes_usage_error(monkeypatch):
    event = {
        "protocol_version": 1,
        "type": "error",
        "reason": "invalid_ref",
        "message": "bad volume ref",
    }
    install_process(
        monkeypatch, FakeProcess(stderr=f"{json.dumps(event)}\n", return_code=2)
    )

    with pytest.raises(click.UsageError, match="invalid_ref"):
        volume_commands.run_cannery(["show", "bad-ref"])


def test_cancellation_is_forwarded_to_child(monkeypatch):
    process = FakeProcess(return_code=None)

    class InterruptingStdout:
        def read(self):
            raise KeyboardInterrupt

    process.stdout = InterruptingStdout()

    def finish_on_signal(value):
        process.signals.append(value)
        process.returncode = 130

    process.send_signal = finish_on_signal
    install_process(monkeypatch, process)

    with pytest.raises(click.Abort):
        volume_commands.run_cannery(["pull", "bdn://dev/model", "/tmp/out"])

    assert process.signals == [signal.SIGINT]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["ls", "models", "--all"], ["ls", "models", "--all"]),
        (["show", "bdn://models/weights"], ["show", "bdn://models/weights"]),
        (
            ["pull", "bdn://models/weights", "output", "--discard"],
            ["pull", "bdn://models/weights", "output", "--discard"],
        ),
    ],
)
def test_volume_commands_are_registered_and_forward_arguments(
    monkeypatch, arguments, expected
):
    run = Mock(return_value={"ok": True})
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(truss_cli, ["volume", *arguments, "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}
    run.assert_called_once_with(expected)


def test_push_command_forwards_optional_ref(monkeypatch, tmp_path):
    source = tmp_path / "weights"
    source.mkdir()
    run = Mock(return_value={"digest": "b3:abc"})
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli,
        ["volume", "push", str(source), "bdn://models/weights:dev", "--output", "json"],
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(["push", str(source), "bdn://models/weights:dev"])


def test_json_output_keeps_status_off_stdout(monkeypatch):
    monkeypatch.setattr(
        volume_commands, "run_cannery", lambda _: {"protocol_version": 1}
    )
    monkeypatch.setattr(
        volume_commands.common,
        "maybe_upgrade_dialogue",
        lambda: volume_commands.console.print("upgrade available"),
    )

    result = CliRunner().invoke(truss_cli, ["volume", "ls", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"protocol_version": 1}
    assert "upgrade available" not in result.stdout
    assert "upgrade available" in result.stderr


def test_volume_help_lists_all_mvp_commands():
    result = CliRunner().invoke(truss_cli, ["volume", "--help"])

    assert result.exit_code == 0
    for command in ("push", "ls", "show", "pull"):
        assert command in result.output
