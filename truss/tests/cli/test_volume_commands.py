import io
import json
import signal
from pathlib import Path
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from truss.cli import volume_commands
from truss.cli.cannery import config as cannery_config
from truss.cli.cannery import runner as cannery_runner
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
    ):
        monkeypatch.delenv(variable, raising=False)
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

    with pytest.raises(click.UsageError, match="non-loopback"):
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


def test_non_loopback_endpoint_rejected_without_token(monkeypatch):
    monkeypatch.setenv("TRUSS_CANNERY_API", "https://cannery.example.com")
    popen = Mock()
    monkeypatch.setattr(volume_commands.subprocess, "Popen", popen)

    with pytest.raises(click.UsageError, match="production token exchange"):
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

    assert "weights/missing" not in str(exc.value)


def test_exit_two_becomes_usage_error(monkeypatch):
    install_process(monkeypatch, FakeProcess(stdout="", return_code=2))

    with pytest.raises(click.UsageError, match="INVALID_ARGUMENT"):
        volume_commands.run_cannery(["show", "bad-ref"])


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
