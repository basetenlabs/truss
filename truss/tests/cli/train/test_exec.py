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
    SUPPORTED_EXEC_ACCELERATORS,
    UV_BASE_IMAGE,
    build_exec_project,
    build_start_commands,
    looks_like_uv_project,
    parse_environment_variables,
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f]")


def _message_text(result) -> str:
    """The CLI output as plain, single-line text.

    rich renders errors and warnings as colorized, width-wrapped panels, and turns
    color on under GITHUB_ACTIONS -- which splits `--flag` tokens with ANSI codes.
    Strip those and the panel borders, and collapse whitespace, so assertions match
    the message regardless of terminal width or color support.
    """
    plain = _BOX_DRAWING_RE.sub(" ", _ANSI_RE.sub("", result.output))
    return " ".join(plain.split())


def _mock_remote():
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = Mock()
    mock_remote.api.get_teams.return_value = {
        "team-a": TeamType(id="team1", name="team-a", default=True)
    }
    mock_remote.api.list_training_projects.return_value = []
    return mock_remote


@contextmanager
def _chdir(directory: Path):
    original_cwd = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _invoke_exec(args, cwd: Path, tail: bool = False):
    """Invoke `truss train exec` from `cwd`, returning (result, mock_push)."""
    base_args = ["train", "exec", "--remote", "test_remote"]
    if not tail:
        base_args.append("--no-tail")

    with (
        _chdir(cwd),
        patch("truss_train.public_api.push") as mock_push,
        patch(
            "truss.cli.train_commands.RemoteFactory.get_remote_team", return_value=None
        ),
        patch(
            "truss.cli.train_commands.RemoteFactory.create", return_value=_mock_remote()
        ),
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
