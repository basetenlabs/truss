import io
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, call

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from truss.cli import volume_commands
from truss.cli.cannery import config as cannery_config
from truss.cli.cannery import runner as cannery_runner
from truss.cli.cannery import v1_protocol
from truss.cli.cannery.errors import CanneryCancelled
from truss.cli.cli import truss_cli

GENERATED_ROOT = Path(volume_commands.__file__).parent / "cannery" / "generated"
FIXTURES = GENERATED_ROOT / "fixtures" / "protojson"
BOOTSTRAP = (
    '{"bootstrap_version":1,"cannery_version":"1.2.3",'
    '"supported_machine_protocols":[1],'
    '"supported_encodings":["protojson-ndjson"]}\n'
)
SUCCESS_FIXTURE = {
    "push": "push-success.ndjson",
    "ls": "list-success.ndjson",
    "show": "show-success.ndjson",
    "pull": "pull-success.ndjson",
}


def _fixture(name):
    return (FIXTURES / name).read_text()


class FakeProcess:
    def __init__(self, stdout=None, stderr="", return_code=0):
        self.default_stdout = stdout is None
        self.stdout = io.StringIO(stdout or "")
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


class _BlockingStderr:
    def __init__(self, closed: threading.Event):
        self._closed = closed

    def read(self, _size):
        self._closed.wait()
        return ""


class HangingAfterTerminalProcess(FakeProcess):
    def __init__(self, stdout):
        super().__init__(stdout=stdout, return_code=None)
        self._closed = threading.Event()
        self.stderr = _BlockingStderr(self._closed)

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if timeout is not None:
            raise subprocess.TimeoutExpired("cannery", timeout)
        self._closed.wait()
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL
        self._closed.set()


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
        "TRUSS_CANNERY_PHASE0",
        "TRUSS_CANNERY_ALLOW_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TRUSS_CANNERY_API", "http://127.0.0.1:8787")
    monkeypatch.setenv("TRUSS_CANNERY_BIN", "/bin/cannery")
    monkeypatch.setenv("TRUSS_CANNERY_DIAGNOSTIC_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(
        cannery_config.RemoteFactory, "get_available_config_names", lambda: []
    )


def install_process(monkeypatch, process):
    def start(argv, **_kwargs):
        if argv == ["/bin/cannery", "protocol"]:
            return FakeProcess(stdout=BOOTSTRAP)
        if process.default_stdout:
            command = argv[argv.index("--api") + 2]
            process.stdout = io.StringIO(_fixture(SUCCESS_FIXTURE[command]))
        return process

    popen = Mock(side_effect=start)
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
    monkeypatch.delenv("TRUSS_CANNERY_BIN")
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
    monkeypatch.setenv("TRUSS_CANNERY_API", "http://127.0.0.1:8787")
    monkeypatch.setenv("TRUSS_CANNERY_ORG", "acme")
    monkeypatch.setenv("TRUSS_CANNERY_AUTH_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("CANNERY_AUTH_TOKEN_FILE", "/must/not/leak")
    popen = install_process(
        monkeypatch, FakeProcess(stdout=_fixture("list-success.ndjson"))
    )

    result = volume_commands.run_cannery(["ls", "models", "--all"])
    correlation_id = result.pop("correlation_id")
    assert result["namespace"] == "weights"
    assert len(result["references"]) == 2
    assert correlation_id == popen.call_args.kwargs["env"]["CANNERY_CORRELATION_ID"]

    argv = popen.call_args.args[0]
    assert argv == [
        "/bin/cannery",
        "--machine-protocol",
        "1",
        "--api",
        "http://127.0.0.1:8787",
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


def test_fresh_install_fails_closed(monkeypatch):
    monkeypatch.delenv("TRUSS_CANNERY_API")
    monkeypatch.delenv("TRUSS_CANNERY_BIN")
    popen = Mock()
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)

    with pytest.raises(click.UsageError, match="truss auth login"):
        volume_commands.run_cannery(["ls"])

    popen.assert_not_called()


def test_v1_bootstrap_mismatch_prevents_operation(monkeypatch):
    unsupported = (
        '{"bootstrap_version":1,"cannery_version":"2.0.0",'
        '"supported_machine_protocols":[2],'
        '"supported_encodings":["protojson-ndjson"]}\n'
    )
    popen = Mock(return_value=FakeProcess(stdout=unsupported))
    monkeypatch.setattr(
        cannery_runner, "resolve_cannery_binary", lambda **_kwargs: "/bin/cannery"
    )
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)

    with pytest.raises(click.ClickException, match="does not support"):
        volume_commands.run_cannery(["ls"])

    popen.assert_called_once()
    assert popen.call_args.args[0] == ["/bin/cannery", "protocol"]


def test_explicit_phase_zero_fallback_is_loopback_only(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUSS_CANNERY_PHASE0", "1")
    process = FakeProcess(stdout='{"protocol_version":1,"refs":[]}')
    popen = install_process(monkeypatch, process)

    result = volume_commands.run_cannery(["ls"])

    assert result["refs"] == []
    assert popen.call_count == 1
    assert popen.call_args.args[0][1:5] == ["-o", "json", "--progress", "machine"]

    token_file = tmp_path / "token"
    token_file.write_text("token")
    token_file.chmod(0o600)
    monkeypatch.setenv("TRUSS_CANNERY_API", "https://cannery.example.com")
    monkeypatch.setenv("TRUSS_CANNERY_AUTH_TOKEN_FILE", str(token_file))
    popen.reset_mock()

    with pytest.raises(click.UsageError, match="loopback"):
        volume_commands.run_cannery(["ls"])

    popen.assert_not_called()


def test_pinned_artifact_rejects_phase_zero_fallback(monkeypatch):
    monkeypatch.setenv("TRUSS_CANNERY_PHASE0", "1")
    popen = install_process(monkeypatch, FakeProcess(stdout='{"protocol_version":1}'))
    monkeypatch.setattr(
        cannery_runner, "binary_diagnostic_metadata", lambda _path: ("1.2.3", "a" * 64)
    )

    with pytest.raises(click.UsageError, match="Pinned Cannery artifacts"):
        volume_commands.run_cannery(["ls"])

    popen.assert_not_called()


def test_non_loopback_endpoint_override_is_rejected(monkeypatch):
    monkeypatch.setenv("TRUSS_CANNERY_API", "https://cannery.example.com")
    popen = Mock()
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)

    with pytest.raises(click.UsageError, match="restricted to an explicit loopback"):
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
    install_process(monkeypatch, FakeProcess(stdout=_fixture("show-success.ndjson")))

    result = volume_commands.run_cannery(["show", "bdn://dev/model"])
    result.pop("correlation_id")
    assert result["manifest_digest"].startswith("b3:")
    assert result["file_page"]["files"]


@pytest.mark.parametrize("stdout", ["", "[]\n", "{}\n", "{}\n{}\n"])
def test_result_parser_rejects_invalid_result_contract(monkeypatch, stdout):
    install_process(monkeypatch, FakeProcess(stdout=stdout))

    with pytest.raises(volume_commands.CanneryProtocolError):
        volume_commands.run_cannery(["ls"])


def test_protocol_mismatch_fails(monkeypatch):
    lines = _fixture("list-success.ndjson").splitlines()
    event = json.loads(lines[0])
    event["protocolVersion"] = 2
    lines[0] = json.dumps(event)
    install_process(monkeypatch, FakeProcess(stdout="\n".join(lines) + "\n"))

    with pytest.raises(
        volume_commands.CanneryProtocolError, match="requires version 1"
    ):
        volume_commands.run_cannery(["ls"])


def test_ndjson_progress_is_drained_and_rendered(monkeypatch, capsys):
    renderer_type = cannery_runner.ProgressRenderer
    monkeypatch.setattr(
        cannery_runner,
        "ProgressRenderer",
        lambda: renderer_type(
            Console(file=sys.stderr, force_terminal=False, color_system=None)
        ),
    )
    install_process(monkeypatch, FakeProcess(stdout=_fixture("push-success.ndjson")))

    volume_commands.run_cannery(["push", "/tmp/model"])

    captured = capsys.readouterr()
    assert "Cannery push (upload)" in captured.err
    assert "4/12 files" in captured.err
    assert "33554432/1073741824 bytes" in captured.err
    assert "COMMITTING_VERSION" in captured.err
    assert "SOURCE_SCAN_RETRIED" in captured.err
    assert captured.out == ""


def test_invalid_ndjson_fails_loudly(monkeypatch):
    install_process(monkeypatch, FakeProcess(stdout="not-json\n"))

    with pytest.raises(volume_commands.CanneryProtocolError, match="invalid ProtoJSON"):
        volume_commands.run_cannery(["ls"])


def test_v1_stderr_is_bounded_redacted_diagnostics_only(monkeypatch):
    secret = "bare-credential-value-9f4c"
    process = FakeProcess(
        stdout="not-json\n",
        stderr="x" * 100_000 + f"\nAuthorization: Bearer {secret}\n",
    )
    install_process(monkeypatch, process)

    with pytest.raises(volume_commands.CanneryProtocolError):
        volume_commands.run_cannery(["ls"])

    diagnostic_dir = Path(cannery_runner.os.environ["TRUSS_CANNERY_DIAGNOSTIC_DIR"])
    diagnostic_text = next(diagnostic_dir.glob("diagnostic-*.jsonl")).read_text()
    assert secret not in diagnostic_text
    assert "[REDACTED]" in diagnostic_text
    assert len(diagnostic_text) < 70_000


def test_machine_error_and_exit_one_become_click_exception(monkeypatch):
    install_process(
        monkeypatch,
        FakeProcess(stdout=_fixture("show-not-found.ndjson"), return_code=1),
    )

    with pytest.raises(click.ClickException, match="reason VOLUME_NOT_FOUND") as exc:
        volume_commands.run_cannery(["show", "bdn://weights/missing:prod"])

    assert "Volume weights/missing was not found" in str(exc.value)


@pytest.mark.parametrize(
    ("fixture_name", "reason", "constraint"),
    [
        (
            "pull-invalid-include.ndjson",
            "INVALID_INCLUDE_PATH",
            "paths must be nonempty",
        ),
        (
            "pull-no-match.ndjson",
            "INCLUDE_PATH_NOT_FOUND",
            "selectors must match at least one file or symlink",
        ),
    ],
)
def test_pull_include_fixture_errors_are_structured_usage_errors(
    monkeypatch, fixture_name, reason, constraint
):
    install_process(
        monkeypatch, FakeProcess(stdout=_fixture(fixture_name), return_code=1)
    )

    with pytest.raises(click.UsageError) as exc_info:
        volume_commands.run_cannery(
            ["pull", "bdn://weights/model:prod", "/tmp/output", "--include", "bad"]
        )

    assert reason in str(exc_info.value)
    assert constraint in str(exc_info.value)


def test_exit_two_becomes_usage_error(monkeypatch):
    install_process(monkeypatch, FakeProcess(stdout="", return_code=2))

    with pytest.raises(click.UsageError, match="INVALID_ARGUMENT"):
        volume_commands.run_cannery(["show", "bad-ref"])


def test_nonempty_pull_destination_error_is_actionable(monkeypatch):
    started = _fixture("pull-integrity-error.ndjson").splitlines()[0]
    error = {
        "protocolVersion": 1,
        "sequence": "2",
        "operationId": "01K0PULL000000000000000002",
        "operation": "OPERATION_PULL",
        "error": {
            "category": "ERROR_CATEGORY_INVALID_ARGUMENT",
            "reason": "INVALID_ARGUMENT",
            "message": "The command arguments are invalid",
            "retryable": False,
            "details": {
                "invalidArgument": {
                    "constraint": "pull destination is not empty: /tmp/output"
                }
            },
        },
    }
    install_process(
        monkeypatch,
        FakeProcess(stdout=f"{started}\n{json.dumps(error)}\n", return_code=1),
    )

    with pytest.raises(click.UsageError) as exc_info:
        volume_commands.run_cannery(["pull", "bdn://weights/model:prod", "/tmp/output"])

    assert "pull destination is not empty: /tmp/output" in str(exc_info.value)


def test_terminal_then_hanging_process_is_killed_within_bound(monkeypatch):
    process = HangingAfterTerminalProcess(_fixture("list-success.ndjson"))
    install_process(monkeypatch, process)
    monkeypatch.setattr(v1_protocol, "_TERMINAL_EXIT_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(cannery_runner, "_CANCEL_GRACE_SECONDS", 0.01)
    started_at = time.monotonic()

    with pytest.raises(
        volume_commands.CanneryProtocolError, match="did not exit within"
    ):
        volume_commands.run_cannery(["ls"])

    assert time.monotonic() - started_at < 1
    assert process.killed
    assert process._closed.is_set()


def test_cancellation_is_forwarded_to_child(monkeypatch):
    process = FakeProcess(return_code=None)

    class InterruptingStdout:
        def readline(self, _size):
            raise KeyboardInterrupt

    process.stdout = InterruptingStdout()
    process.default_stdout = False

    def finish_on_signal(value):
        process.signals.append(value)
        process.returncode = 130

    process.send_signal = finish_on_signal
    install_process(monkeypatch, process)

    with pytest.raises(CanneryCancelled) as exc_info:
        volume_commands.run_cannery(["pull", "bdn://dev/model", "/tmp/out"])

    assert exc_info.value.exit_code == 130
    assert process.signals == [signal.SIGINT]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["ls", "models", "--all"], ["ls", "models", "--all"]),
        (["show", "bdn://models/weights"], ["show", "bdn://models/weights"]),
        (
            ["pull", "bdn://models/weights", "output"],
            ["pull", "bdn://models/weights", "output"],
        ),
    ],
)
def test_volume_commands_are_registered_and_forward_arguments(
    monkeypatch, arguments, expected
):
    if arguments[0] == "ls":
        command_result = {"namespaces": []}
    elif arguments[0] == "show":
        command_result = {
            "manifest_digest": "b3:abc",
            "canonical_reference": arguments[1],
            "file_page": {"files": []},
        }
    else:
        command_result = {"ok": True}
    run = Mock(return_value=command_result)
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(truss_cli, ["volume", *arguments, "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == command_result
    run.assert_called_once_with(expected, remote=None)


def test_pull_command_forwards_repeated_includes_then_restart(monkeypatch):
    run = Mock(return_value={"content_verified": True})
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli,
        [
            "volume",
            "pull",
            "bdn://models/weights",
            "output",
            "--include",
            "model.safetensors",
            "--include",
            "tokenizer/",
            "--restart",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        [
            "pull",
            "bdn://models/weights",
            "output",
            "--include",
            "model.safetensors",
            "--include",
            "tokenizer/",
            "--restart",
        ],
        remote=None,
    )


def test_pull_command_rejects_empty_include_before_cannery(monkeypatch):
    run = Mock()
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli, ["volume", "pull", "bdn://models/weights", "output", "--include", ""]
    )

    assert result.exit_code != 0
    assert "must not be empty" in result.output
    run.assert_not_called()


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
    run.assert_called_once_with(
        ["push", str(source), "bdn://models/weights:dev"], remote=None
    )


def test_json_output_keeps_status_off_stdout(monkeypatch):
    monkeypatch.setattr(
        volume_commands,
        "run_cannery",
        lambda _arguments, remote=None: {"namespaces": [], "protocol_version": 1},
    )
    monkeypatch.setattr(
        volume_commands.common,
        "maybe_upgrade_dialogue",
        lambda: volume_commands.console.print("upgrade available"),
    )

    result = CliRunner().invoke(truss_cli, ["volume", "ls", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"namespaces": [], "protocol_version": 1}
    assert "upgrade available" not in result.stdout
    assert "upgrade available" in result.stderr


@pytest.mark.parametrize("output_format", ["json", "text"])
def test_pull_output_preserves_selected_and_volume_totals(monkeypatch, output_format):
    command_result = {
        "logical_bytes": "268435456",
        "downloaded_bytes": "201326592",
        "reused_bytes": "67108864",
        "file_count": "3",
        "directory_count": "1",
        "volume_logical_bytes": "1073741824",
        "volume_file_count": "12",
        "volume_directory_count": "3",
    }
    monkeypatch.setattr(
        volume_commands, "run_cannery", Mock(return_value=command_result)
    )
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli,
        ["volume", "pull", "bdn://weights/model", "output", "--output", output_format],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == command_result


def test_remote_option_is_forwarded(monkeypatch):
    run = Mock(return_value={"namespaces": []})
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli, ["volume", "ls", "--remote", "staging", "--output", "json"]
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(["ls"], remote="staging")


def test_list_consumes_all_metadata_pages(monkeypatch):
    run = Mock(
        side_effect=[
            {
                "namespace": "weights",
                "references": [{"reference": "first"}],
                "next_page_token": "p2",
            },
            {"namespace": "weights", "references": [{"reference": "second"}]},
        ]
    )
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli,
        ["volume", "ls", "weights", "--page-size", "1000", "--output", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["references"] == [
        {"reference": "first"},
        {"reference": "second"},
    ]
    assert run.call_args_list == [
        call(["ls", "weights", "--page-size", "1000"], remote=None),
        call(
            ["ls", "weights", "--page-size", "1000", "--page-token", "p2"], remote=None
        ),
    ]


def test_show_rejects_repeated_page_token(monkeypatch):
    run = Mock(
        side_effect=[
            {
                "manifest_digest": "b3:abc",
                "canonical_reference": "bdn://weights/model@b3:abc",
                "file_page": {"files": [], "next_page_token": "repeat"},
            },
            {
                "manifest_digest": "b3:abc",
                "canonical_reference": "bdn://weights/model@b3:abc",
                "file_page": {"files": [], "next_page_token": "repeat"},
            },
        ]
    )
    monkeypatch.setattr(volume_commands, "run_cannery", run)
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli, ["volume", "show", "bdn://weights/model", "--output", "json"]
    )

    assert result.exit_code != 0
    assert "repeated" in result.output
    assert "next_page_token" in result.output


def test_cli_cancellation_exits_130(monkeypatch, tmp_path):
    monkeypatch.setattr(
        volume_commands, "run_cannery", Mock(side_effect=CanneryCancelled())
    )
    monkeypatch.setattr(volume_commands.common, "maybe_upgrade_dialogue", lambda: None)

    result = CliRunner().invoke(
        truss_cli, ["volume", "pull", "bdn://weights/model", str(tmp_path / "out")]
    )

    assert result.exit_code == 130


def test_volume_help_lists_all_mvp_commands():
    result = CliRunner().invoke(truss_cli, ["volume", "--help"])

    assert result.exit_code == 0
    for command in ("push", "ls", "show", "pull"):
        assert command in result.output


def test_pull_help_exposes_resume_controls_without_discard():
    result = CliRunner().invoke(truss_cli, ["volume", "pull", "--help"])

    assert result.exit_code == 0
    assert "--include" in result.output
    assert "--restart" in result.output
    assert "--discard" not in result.output
