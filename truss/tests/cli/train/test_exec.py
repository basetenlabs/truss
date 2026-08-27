import os
import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import rich_click as click
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.cli.train.exec import (
    DEFAULT_CPU_COUNT,
    DEFAULT_MEMORY,
    PYTHON_BASE_IMAGE,
    SECRETS_SETTINGS_URL,
    SUPPORTED_EXEC_ACCELERATORS,
    UV_BASE_IMAGE,
    build_exec_project,
    build_start_commands,
    looks_like_uv_project,
    parse_environment_variables,
    warn_about_missing_secrets,
)
from truss.cli.train.workstation import DEFAULT_BASE_IMAGE
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote
from truss_train.definitions import (
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
    SecretReference,
)

USER_COMMAND = ["uv", "run", "python", "my_script.py"]
USER_COMMAND_STR = "uv run python my_script.py"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f]")


def _plain(text: str) -> str:
    """`text` as plain, single-line text.

    rich renders errors and warnings as colorized, width-wrapped panels, and turns
    color on under GITHUB_ACTIONS (and FORCE_COLOR) -- which splits tokens like
    `--flag` with ANSI codes. Strip those and the panel borders, and collapse
    whitespace, so assertions match the message regardless of terminal width or
    color support.
    """
    return " ".join(_BOX_DRAWING_RE.sub(" ", _ANSI_RE.sub("", text)).split())


def _message_text(result) -> str:
    """The CLI result's output as plain, single-line text."""
    return _plain(result.output)


def _uv_project(tmp_path: Path, lock: bool = True) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    if lock:
        (tmp_path / "uv.lock").write_text("")
    return tmp_path


# --- builder: compute, image, session, workspace -----------------------------


def test_build_exec_project_cpu_defaults(tmp_path):
    project = build_exec_project(
        start_command=["python", "my_script.py"], project_name="my-project"
    )
    assert project.name == "my-project"

    job = project.job
    assert job.compute.accelerator is None
    assert job.compute.cpu_count == DEFAULT_CPU_COUNT
    assert job.compute.memory == DEFAULT_MEMORY
    assert job.compute.node_count == 1
    assert job.image.base_image == PYTHON_BASE_IMAGE
    assert job.workspace is None


def test_build_exec_project_gpu(tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        accelerator="H100",
        gpu_count=4,
    ).job
    assert job.compute.accelerator is not None
    assert job.compute.accelerator.accelerator.value == "H100"
    assert job.compute.accelerator.count == 4
    assert job.image.base_image == DEFAULT_BASE_IMAGE


@pytest.mark.parametrize("accelerator", SUPPORTED_EXEC_ACCELERATORS)
def test_build_exec_project_supported_accelerators(accelerator, tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        accelerator=accelerator,
    ).job
    assert job.compute.accelerator.accelerator.value == accelerator


def test_build_exec_project_invalid_accelerator(tmp_path):
    with pytest.raises(ValueError):
        build_exec_project(
            start_command=["python", "my_script.py"],
            project_name="my-project",
            accelerator="INVALID",
        )


def test_build_exec_project_enables_ssh_on_demand(tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"], project_name="my-project"
    ).job
    assert job.interactive_session is not None
    assert job.interactive_session.trigger == InteractiveSessionTrigger.ON_DEMAND
    assert job.interactive_session.session_provider == InteractiveSessionProvider.SSH


def test_build_exec_project_leaves_storage_disabled(tmp_path):
    runtime = build_exec_project(
        start_command=["python", "my_script.py"], project_name="my-project"
    ).job.runtime
    assert runtime.cache_config is None
    assert runtime.load_checkpoint_config is None
    assert runtime.checkpointing_config.enabled is False
    assert runtime.checkpointing_config.checkpoint_path is None


def test_build_exec_project_custom_image_wins(tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        base_image="my-registry/my-image:latest",
    ).job
    assert job.image.base_image == "my-registry/my-image:latest"


def test_build_exec_project_workspace_from_dir_flags(tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        workspace_root="..",
        exclude_dirs=["data", "checkpoints"],
        external_dirs=["../shared"],
    ).job
    assert job.workspace is not None
    assert job.workspace.workspace_root == ".."
    assert job.workspace.exclude_dirs == ["data", "checkpoints"]
    assert job.workspace.external_dirs == ["../shared"]


def test_build_exec_project_environment_variables(tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        environment_variables={
            "PLAIN": "1",
            "BASETEN_API_KEY": SecretReference(name="my_api_key"),
        },
    ).job
    assert job.runtime.environment_variables == {
        "PLAIN": "1",
        "BASETEN_API_KEY": SecretReference(name="my_api_key"),
    }


def test_build_exec_project_no_environment_variables_by_default(tmp_path):
    job = build_exec_project(
        start_command=["python", "my_script.py"], project_name="my-project"
    ).job
    assert job.runtime.environment_variables == {}


# --- builder: --env / --secret parsing ---------------------------------------


def test_parse_environment_variables_splits_on_first_equals_only():
    parsed = parse_environment_variables(env=["TOKEN=abc=def==", "PLAIN=1"])
    assert parsed == {"TOKEN": "abc=def==", "PLAIN": "1"}


def test_parse_environment_variables_builds_secret_references():
    parsed = parse_environment_variables(secrets=["BASETEN_API_KEY=my_api_key"])
    assert parsed == {"BASETEN_API_KEY": SecretReference(name="my_api_key")}
    assert isinstance(parsed["BASETEN_API_KEY"], SecretReference)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"env": ["NO_EQUALS"]}, "Invalid --env value 'NO_EQUALS'"),
        ({"env": ["=novalue"]}, "Invalid --env value '=novalue'"),
        ({"secrets": ["NO_EQUALS"]}, "Invalid --secret value 'NO_EQUALS'"),
    ],
)
def test_parse_environment_variables_rejects_entries_without_a_key(kwargs, expected):
    with pytest.raises(click.UsageError, match=re.escape(expected)):
        parse_environment_variables(**kwargs)


def test_parse_environment_variables_rejects_duplicate_keys():
    with pytest.raises(click.UsageError, match="set more than once"):
        parse_environment_variables(env=["KEY=literal"], secrets=["KEY=secret_name"])


# --- builder: uv detection and start commands --------------------------------


@pytest.mark.parametrize("filename", ["uv.lock", "pyproject.toml"])
def test_looks_like_uv_project(filename, tmp_path):
    assert not looks_like_uv_project(tmp_path)
    (tmp_path / filename).write_text("")
    assert looks_like_uv_project(tmp_path)


def test_build_start_commands_runs_the_command_verbatim():
    assert build_start_commands(
        start_command=["python", "my script.py", "--steps", "100"]
    ) == ["python 'my script.py' --steps 100"]


def test_build_start_commands_prepends_the_idempotent_uv_install():
    assert build_start_commands(start_command=USER_COMMAND, install_uv=True) == [
        "/bin/sh -c '{ command -v uv >/dev/null 2>&1 || "
        "curl -LsSf https://astral.sh/uv/install.sh | sh ; } && "
        'export PATH="$HOME/.local/bin:$PATH" && '
        f"{USER_COMMAND_STR}'"
    ]


def test_build_exec_project_with_uv_selects_the_uv_image_on_cpu():
    job = build_exec_project(
        start_command=USER_COMMAND, project_name="my-project", with_uv=True
    ).job
    assert job.image.base_image == UV_BASE_IMAGE
    # The uv image already ships uv, so the command is the only start command.
    assert job.runtime.start_commands == [USER_COMMAND_STR]


def test_build_exec_project_with_uv_installs_uv_on_the_gpu_image():
    job = build_exec_project(
        start_command=USER_COMMAND,
        project_name="my-project",
        accelerator="H100",
        with_uv=True,
    ).job
    assert job.image.base_image == DEFAULT_BASE_IMAGE
    assert "astral.sh/uv/install.sh" in job.runtime.start_commands[0]
    assert job.runtime.start_commands[0].endswith(f"{USER_COMMAND_STR}'")


def test_build_exec_project_with_uv_installs_uv_on_a_custom_image():
    job = build_exec_project(
        start_command=USER_COMMAND,
        project_name="my-project",
        base_image="my-registry/my-image:latest",
        with_uv=True,
    ).job
    assert job.image.base_image == "my-registry/my-image:latest"
    assert "astral.sh/uv/install.sh" in job.runtime.start_commands[0]
    assert job.runtime.start_commands[0].endswith(f"{USER_COMMAND_STR}'")


@pytest.mark.parametrize("accelerator", [None, "H100"])
def test_build_exec_project_without_with_uv_injects_nothing(accelerator):
    job = build_exec_project(
        start_command=USER_COMMAND, project_name="my-project", accelerator=accelerator
    ).job
    assert job.runtime.start_commands == [USER_COMMAND_STR]


# --- CLI ---------------------------------------------------------------------


def _mock_remote(secrets=("my_api_key",)):
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = Mock()
    mock_remote.api.get_teams.return_value = {
        "team-a": TeamType(id="team1", name="team-a", default=True)
    }
    mock_remote.api.list_training_projects.return_value = []
    mock_remote.api.get_all_secrets.return_value = {
        "secrets": [{"name": name} for name in secrets]
    }
    return mock_remote


@contextmanager
def _chdir(directory: Path):
    original_cwd = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _invoke_exec(args, cwd: Path, tail: bool = False, remote=None):
    """Invoke `truss train exec` from `cwd`, returning (result, mock_push)."""
    base_args = ["train", "exec", "--remote", "test_remote"]
    if not tail:
        base_args.append("--no-tail")
    remote = remote if remote is not None else _mock_remote()

    with (
        _chdir(cwd),
        patch("truss_train.public_api.push") as mock_push,
        patch(
            "truss.cli.train_commands.RemoteFactory.get_remote_team", return_value=None
        ),
        patch("truss.cli.train_commands.RemoteFactory.create", return_value=remote),
    ):
        mock_push.return_value = {
            "id": "job123",
            "training_project": {"id": "proj123", "name": cwd.name},
        }
        result = CliRunner().invoke(truss_cli, base_args + list(args))

    return result, mock_push


def test_exec_passes_start_command_through_after_double_dash(tmp_path):
    result, mock_push = _invoke_exec(
        ["--", "python", "my_script.py", "--steps", "100", "--verbose"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["config"].job.runtime.start_commands == [
        "python my_script.py --steps 100 --verbose"
    ]


def test_exec_start_command_may_reuse_our_own_flag_names(tmp_path):
    """Everything after `--` belongs to the command, even `--memory`/`--tail`."""
    result, mock_push = _invoke_exec(
        [
            "--memory",
            "16Gi",
            "--",
            "python",
            "my_script.py",
            "--memory",
            "4Gi",
            "--tail",
        ],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    job = mock_push.call_args[1]["config"].job
    assert job.runtime.start_commands == ["python my_script.py --memory 4Gi --tail"]
    assert job.compute.memory == "16Gi"


def test_exec_defaults_to_cpu_only_job(tmp_path):
    result, mock_push = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    job = mock_push.call_args[1]["config"].job
    assert job.compute.accelerator is None
    assert job.compute.cpu_count == DEFAULT_CPU_COUNT
    assert job.compute.memory == DEFAULT_MEMORY


def test_exec_pushes_current_directory_as_source_dir(tmp_path):
    result, mock_push = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["source_dir"] == tmp_path.resolve()


def test_exec_defaults_project_name_to_directory_name(tmp_path):
    work_dir = tmp_path / "my-checkout"
    work_dir.mkdir()

    result, mock_push = _invoke_exec(["--", "python", "my_script.py"], work_dir)

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["config"].name == "my-checkout"


def test_exec_project_name_overrides_directory_name(tmp_path):
    result, mock_push = _invoke_exec(
        ["--project-name", "explicit-name", "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["config"].name == "explicit-name"


def test_exec_requires_a_start_command(tmp_path):
    result, mock_push = _invoke_exec([], tmp_path)

    assert result.exit_code != 0
    assert "No start command given" in _message_text(result)
    mock_push.assert_not_called()


def test_exec_gpu_count_requires_accelerator(tmp_path):
    result, mock_push = _invoke_exec(
        ["--gpu-count", "2", "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code != 0
    assert "--gpu-count requires --accelerator" in _message_text(result)
    mock_push.assert_not_called()


def test_exec_accelerator_is_normalized(tmp_path):
    result, mock_push = _invoke_exec(
        ["--accelerator", "h100", "--gpu-count", "2", "--", "python", "my_script.py"],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    accelerator = mock_push.call_args[1]["config"].job.compute.accelerator
    assert accelerator.accelerator.value == "H100"
    assert accelerator.count == 2


def test_exec_cpu_and_memory_flags(tmp_path):
    result, mock_push = _invoke_exec(
        ["--cpu-count", "8", "--memory", "16Gi", "--", "python", "my_script.py"],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    compute = mock_push.call_args[1]["config"].job.compute
    assert compute.cpu_count == 8
    assert compute.memory == "16Gi"


def test_exec_directory_flags_build_a_workspace(tmp_path):
    result, mock_push = _invoke_exec(
        [
            "--workspace-root",
            "..",
            "--exclude-dir",
            "data",
            "--exclude-dir",
            "logs",
            "--",
            "python",
            "my_script.py",
        ],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    workspace = mock_push.call_args[1]["config"].job.workspace
    assert workspace is not None
    assert workspace.workspace_root == ".."
    assert workspace.exclude_dirs == ["data", "logs"]


def test_exec_passes_team_id_to_push(tmp_path):
    result, mock_push = _invoke_exec(
        ["--team", "team-a", "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["team_id"] == "team1"


def test_exec_env_and_secret_flags(tmp_path):
    result, mock_push = _invoke_exec(
        [
            "--env",
            "MY_URL=https://example.com/?a=1&b=2",
            "--secret",
            "BASETEN_API_KEY=my_api_key",
            "--",
            "python",
            "my_script.py",
        ],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["config"].job.runtime.environment_variables == {
        "MY_URL": "https://example.com/?a=1&b=2",
        "BASETEN_API_KEY": SecretReference(name="my_api_key"),
    }


def test_exec_rejects_env_without_equals(tmp_path):
    result, mock_push = _invoke_exec(
        ["--env", "BROKEN", "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code != 0
    assert "Invalid --env value" in _message_text(result)
    mock_push.assert_not_called()


def test_exec_rejects_key_set_by_both_env_and_secret(tmp_path):
    result, mock_push = _invoke_exec(
        [
            "--env",
            "KEY=literal",
            "--secret",
            "KEY=secret_name",
            "--",
            "python",
            "my_script.py",
        ],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "set more than once" in _message_text(result)
    mock_push.assert_not_called()


def test_exec_with_uv_selects_the_uv_image_and_keeps_the_command(tmp_path):
    result, mock_push = _invoke_exec(
        ["--with-uv", "--"] + USER_COMMAND, _uv_project(tmp_path)
    )

    assert result.exit_code == 0, result.output
    job = mock_push.call_args[1]["config"].job
    assert job.image.base_image == UV_BASE_IMAGE
    assert job.runtime.start_commands == [USER_COMMAND_STR]


def test_exec_without_with_uv_uses_the_plain_python_image(tmp_path):
    result, mock_push = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    job = mock_push.call_args[1]["config"].job
    assert job.image.base_image == PYTHON_BASE_IMAGE
    assert job.runtime.start_commands == ["python my_script.py"]


@pytest.mark.parametrize("with_uv", [True, False])
def test_exec_image_flag_overrides_the_base_image(with_uv, tmp_path):
    args = ["--image", "my-registry/my-image:latest", "--"] + USER_COMMAND
    if with_uv:
        args.insert(0, "--with-uv")

    result, mock_push = _invoke_exec(args, _uv_project(tmp_path))

    assert result.exit_code == 0, result.output
    job = mock_push.call_args[1]["config"].job
    assert job.image.base_image == "my-registry/my-image:latest"
    if with_uv:
        # An unknown image may not ship uv, so the install step is prepended.
        assert "astral.sh/uv/install.sh" in job.runtime.start_commands[0]
        assert job.runtime.start_commands[0].endswith(f"{USER_COMMAND_STR}'")
    else:
        assert job.runtime.start_commands == [USER_COMMAND_STR]


@pytest.mark.parametrize("filename", ["uv.lock", "pyproject.toml"])
def test_exec_warns_about_a_uv_project_without_with_uv(filename, tmp_path):
    (tmp_path / filename).write_text("")

    result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" in _message_text(result)


def test_exec_does_not_warn_when_with_uv_is_passed(tmp_path):
    result, _ = _invoke_exec(["--with-uv", "--"] + USER_COMMAND, _uv_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" not in _message_text(result)


def test_exec_does_not_warn_for_a_plain_directory(tmp_path):
    result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" not in _message_text(result)


# --- secret existence warning ---------------------------------------------------


def _api(secrets_response):
    api = Mock()
    api.get_all_secrets.return_value = secrets_response
    return api


def test_warn_about_missing_secrets_skips_the_call_without_any_secret_refs(capsys):
    api = _api({"secrets": []})

    warn_about_missing_secrets(api, {"PLAIN": "literal"})

    api.get_all_secrets.assert_not_called()
    assert _plain(capsys.readouterr().out) == ""


def test_warn_about_missing_secrets_is_quiet_when_the_secret_exists(capsys):
    api = _api({"secrets": [{"name": "my_api_key"}]})

    warn_about_missing_secrets(api, {"K": SecretReference(name="my_api_key")})

    api.get_all_secrets.assert_called_once_with()
    assert _plain(capsys.readouterr().out) == ""


def test_warn_about_missing_secrets_warns_when_the_secret_is_absent(capsys):
    api = _api({"secrets": [{"name": "other"}]})

    warn_about_missing_secrets(api, {"K": SecretReference(name="my_api_key")})

    out = _plain(capsys.readouterr().out)
    assert "Warning:" in out
    assert "no secret named my_api_key found in this workspace" in out
    assert "Create it at" in out
    assert SECRETS_SETTINGS_URL in out


def test_warn_about_missing_secrets_warns_for_an_empty_workspace(capsys):
    """An empty listing is a real answer, unlike an unreadable one."""
    warn_about_missing_secrets(
        _api({"secrets": []}), {"K": SecretReference(name="my_api_key")}
    )

    assert "my_api_key" in _plain(capsys.readouterr().out)


def test_warn_about_missing_secrets_names_every_missing_secret(capsys):
    warn_about_missing_secrets(
        _api({"secrets": [{"name": "present"}]}),
        {
            "A": SecretReference(name="missing_a"),
            "B": SecretReference(name="present"),
            "C": SecretReference(name="missing_b"),
            "D": "literal",
        },
    )

    out = _plain(capsys.readouterr().out)
    assert "missing_a" in out and "missing_b" in out
    assert "present" not in out
    # Plural form, since more than one name is listed.
    assert "no secrets named" in out and "Create them at" in out


@pytest.mark.parametrize(
    "response",
    [
        ["my_api_key"],
        [{"name": "my_api_key"}],
        {"secrets": ["my_api_key"]},
        {"secrets": [{"name": "my_api_key"}]},
    ],
)
def test_warn_about_missing_secrets_accepts_the_plausible_payload_shapes(
    response, capsys
):
    warn_about_missing_secrets(
        _api(response), {"K": SecretReference(name="my_api_key")}
    )

    assert _plain(capsys.readouterr().out) == ""


@pytest.mark.parametrize(
    "response", ["a string", 42, None, {"data": []}, {"secrets": "nope"}, [123]]
)
def test_warn_about_missing_secrets_stays_silent_on_an_unreadable_payload(
    response, capsys
):
    """Rather than guess a shape and warn about a secret that is really there."""
    warn_about_missing_secrets(
        _api(response), {"K": SecretReference(name="my_api_key")}
    )

    assert _plain(capsys.readouterr().out) == ""


def test_warn_about_missing_secrets_swallows_api_errors(capsys):
    api = Mock()
    api.get_all_secrets.side_effect = RuntimeError("403 Forbidden")

    warn_about_missing_secrets(api, {"K": SecretReference(name="my_api_key")})

    assert _plain(capsys.readouterr().out) == ""


def test_warn_about_missing_secrets_escapes_console_markup(capsys):
    warn_about_missing_secrets(
        _api({"secrets": []}), {"K": SecretReference(name="[bold]weird")}
    )

    assert "[bold]weird" in _plain(capsys.readouterr().out)


def test_exec_warns_but_still_pushes_when_a_secret_is_missing(tmp_path):
    remote = _mock_remote(secrets=("some_other_secret",))

    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=remote,
    )

    assert result.exit_code == 0, result.output
    assert "no secret named my_api_key found in this workspace" in _message_text(result)
    assert SECRETS_SETTINGS_URL in _message_text(result)
    mock_push.assert_called_once()


def test_exec_does_not_warn_when_the_secret_exists(tmp_path):
    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=_mock_remote(secrets=("my_api_key",)),
    )

    assert result.exit_code == 0, result.output
    assert "no secret named" not in _message_text(result)
    mock_push.assert_called_once()


def test_exec_still_pushes_when_listing_secrets_fails(tmp_path):
    remote = _mock_remote()
    remote.api.get_all_secrets.side_effect = RuntimeError("403 Forbidden")

    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=remote,
    )

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert "403 Forbidden" not in result.output
    mock_push.assert_called_once()


def test_exec_skips_the_secrets_call_without_secret_flags(tmp_path):
    remote = _mock_remote()

    result, mock_push = _invoke_exec(
        ["--env", "PLAIN=1", "--", "python", "my_script.py"], tmp_path, remote=remote
    )

    assert result.exit_code == 0, result.output
    remote.api.get_all_secrets.assert_not_called()
    mock_push.assert_called_once()


def test_exec_tails_logs_by_default(tmp_path):
    with patch("truss.cli.train_commands.TrainingLogWatcher") as mock_watcher:
        mock_watcher.return_value.watch.return_value = []
        result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path, tail=True)

    assert result.exit_code == 0, result.output
    assert mock_watcher.call_args[0][1:] == ("proj123", "job123")


def test_exec_no_tail_skips_the_log_watcher(tmp_path):
    with patch("truss.cli.train_commands.TrainingLogWatcher") as mock_watcher:
        result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    mock_watcher.assert_not_called()
