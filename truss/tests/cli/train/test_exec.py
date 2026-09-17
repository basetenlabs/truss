import inspect
import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import rich_click as click
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.cli.loops_commands import LOOPS_EXEC_CPU_COUNT, LOOPS_EXEC_MEMORY
from truss.cli.train.exec import (
    API_KEYS_SETTINGS_URL,
    BASETEN_API_KEY_ENV_VAR,
    PYTHON_BASE_IMAGE,
    SECRETS_SETTINGS_URL,
    SUPPORTED_EXEC_ACCELERATORS,
    OrphanedApiKeyError,
    UvProject,
    build_exec_project,
    build_start_commands,
    default_base_image,
    ensure_team_api_key_secret,
    get_project_type,
    parse_environment_variables,
    resolve_workspace_root,
    team_api_key_secret_name,
    validate_secret_references,
    validate_workspace_root,
)
from truss.cli.train.exec.builder import (
    JOB_WORKING_DIR,
    UV_CACHE_DIR,
    UV_CACHE_DIR_ENV_VAR,
)
from truss.cli.train.exec.uv import (
    PYPROJECT_FILE,
    UV_BASE_IMAGE,
    UV_CURL_INSTALL,
    UV_INSTALL_STEPS,
    UV_LOCK_FILE,
    UV_PIP_INSTALL,
    UV_PRESENT_PROBE,
    is_uv_project,
)
from truss.cli.train.workstation import DEFAULT_BASE_IMAGE
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.custom_types import APIKeyCategory, TeamType
from truss.remote.baseten.remote import BasetenRemote
from truss_train.definitions import (
    CacheConfig,
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
    SecretReference,
)

USER_COMMAND = ["uv", "run", "python", "my_script.py"]
USER_COMMAND_STR = "uv run python my_script.py"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f]")


def _strip_ansi(text: str) -> str:
    """`text` with terminal escapes removed but its structure intact."""
    return _ANSI_RE.sub("", text)


def _split_streams_runner() -> CliRunner:
    """A runner whose `result.stdout` carries stdout and nothing else.

    click below 8.2 folds stderr into stdout unless told otherwise; 8.2 and later
    always keep the two apart and dropped the argument. Tests that assert stdout
    holds only the JSON payload need the split on both, and the declared floor
    (click>=8.0.3) is exercised by the lowest-direct CI job.
    """
    if "mix_stderr" in inspect.signature(CliRunner.__init__).parameters:
        return CliRunner(mix_stderr=False)
    return CliRunner()


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


def _build(**overrides):
    """Build with a complete argument set, so each test overrides only what it
    exercises. `build_exec_project` takes no defaults of its own -- the CLI is the
    single source of those -- so a baseline has to live somewhere, and a test helper
    is the right place for it."""
    kwargs = dict(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        accelerator=None,
        gpu_count=1,
        cpu_count=LOOPS_EXEC_CPU_COUNT,
        memory=LOOPS_EXEC_MEMORY,
        base_image=None,
        project=None,
        workspace_root=None,
        exclude_dirs=(),
        external_dirs=(),
        environment_variables={},
        enable_cache=True,
    )
    kwargs.update(overrides)
    return build_exec_project(**kwargs)


def test_build_exec_project_requires_every_argument():
    """The looseness this guards against: a caller half-specifying a project."""
    with pytest.raises(TypeError):
        build_exec_project(start_command=["python", "x.py"], project_name="p")


def test_build_exec_project_is_keyword_only():
    with pytest.raises(TypeError):
        build_exec_project(["python", "x.py"], "p")  # type: ignore[misc]


# --- project types -----------------------------------------------------------


def test_get_project_type_detects_uv(tmp_path):
    (tmp_path / UV_LOCK_FILE).write_text("")
    project = get_project_type(tmp_path)
    assert isinstance(project, UvProject)
    assert project.label == "uv"


def test_get_project_type_returns_none_for_an_unrecognised_directory(tmp_path):
    assert get_project_type(tmp_path) is None


def test_uv_project_wants_the_uv_image():
    assert UvProject().base_image() == UV_BASE_IMAGE


def test_uv_project_setup_is_empty_on_the_uv_image():
    """The image already ships uv, so there is nothing to prepend."""
    assert UvProject().setup(UV_BASE_IMAGE) == []


@pytest.mark.parametrize("image", ["nvidia/cuda:12.8.1-devel-ubuntu24.04", "custom:1"])
def test_uv_project_setup_installs_uv_on_any_other_image(image):
    steps = UvProject().setup(image)
    assert steps == UV_INSTALL_STEPS
    assert "astral.sh/uv/install.sh" in steps[0]


def test_default_base_image_prefers_the_accelerator_over_the_project():
    """A GPU job needs the CUDA image; the project's setup steps cover the gap."""
    assert default_base_image("H100", UvProject()) == DEFAULT_BASE_IMAGE
    assert default_base_image(None, UvProject()) == UV_BASE_IMAGE
    assert default_base_image(None, None) == PYTHON_BASE_IMAGE


# --- builder: compute, image, session, workspace -----------------------------


def test_build_exec_project_cpu_defaults(tmp_path):
    project = _build(
        start_command=["python", "my_script.py"], project_name="my-project"
    )
    assert project.name == "my-project"

    job = project.job
    assert job.compute.accelerator is None
    assert job.compute.cpu_count == LOOPS_EXEC_CPU_COUNT
    assert job.compute.memory == LOOPS_EXEC_MEMORY
    assert job.compute.node_count == 1
    assert job.image.base_image == PYTHON_BASE_IMAGE
    assert job.workspace is None


def test_build_exec_project_gpu(tmp_path):
    job = _build(
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
    job = _build(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        accelerator=accelerator,
    ).job
    assert job.compute.accelerator.accelerator.value == accelerator


def test_build_exec_project_invalid_accelerator(tmp_path):
    with pytest.raises(ValueError):
        _build(
            start_command=["python", "my_script.py"],
            project_name="my-project",
            accelerator="INVALID",
        )


def test_build_exec_project_enables_ssh_on_demand(tmp_path):
    job = _build(
        start_command=["python", "my_script.py"], project_name="my-project"
    ).job
    assert job.interactive_session is not None
    assert job.interactive_session.trigger == InteractiveSessionTrigger.ON_DEMAND
    assert job.interactive_session.session_provider == InteractiveSessionProvider.SSH


def test_build_exec_project_leaves_checkpointing_disabled(tmp_path):
    """A one-off command has no checkpoints to write or resume from; the cache
    volume it does get is covered separately."""
    runtime = _build(
        start_command=["python", "my_script.py"], project_name="my-project"
    ).job.runtime
    assert runtime.load_checkpoint_config is None
    assert runtime.checkpointing_config.enabled is False
    assert runtime.checkpointing_config.checkpoint_path is None


def test_build_exec_project_custom_image_wins(tmp_path):
    job = _build(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        base_image="my-registry/my-image:latest",
    ).job
    assert job.image.base_image == "my-registry/my-image:latest"


def test_build_exec_project_workspace_from_dir_flags(tmp_path):
    job = _build(
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
    job = _build(
        start_command=["python", "my_script.py"],
        project_name="my-project",
        environment_variables={
            "PLAIN": "1",
            "BASETEN_API_KEY": SecretReference(name="my_api_key"),
        },
    ).job
    assert job.runtime.environment_variables == {
        UV_CACHE_DIR_ENV_VAR: UV_CACHE_DIR,
        "PLAIN": "1",
        "BASETEN_API_KEY": SecretReference(name="my_api_key"),
    }


def test_build_exec_project_sets_only_the_uv_cache_by_default(tmp_path):
    """Nothing else is injected; the uv cache is a placement fix, not a policy."""
    job = _build(
        start_command=["python", "my_script.py"], project_name="my-project"
    ).job
    assert job.runtime.environment_variables == {UV_CACHE_DIR_ENV_VAR: UV_CACHE_DIR}


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


def test_build_start_commands_runs_the_command_verbatim():
    assert build_start_commands(
        start_command=["python", "my script.py", "--steps", "100"]
    ) == ["python 'my script.py' --steps 100"]


def test_build_start_commands_prepends_the_idempotent_uv_install():
    assert build_start_commands(
        start_command=USER_COMMAND, setup_steps=UV_INSTALL_STEPS
    ) == [
        "/bin/sh -c '{ command -v uv >/dev/null 2>&1 || "
        "pip install --quiet uv || { "
        "curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh && "
        "sh /tmp/uv-install.sh ; } ; } && "
        'export PATH="$HOME/.local/bin:$PATH" && '
        f"{USER_COMMAND_STR}'"
    ]


# --- uv cache placement ------------------------------------------------------


def test_uv_cache_is_set_in_the_job_environment():
    """A shell export would not reach an interactive session into the job, which
    this command enables by default."""
    env = _build().job.runtime.environment_variables
    assert env[UV_CACHE_DIR_ENV_VAR] == UV_CACHE_DIR
    assert UV_CACHE_DIR.startswith(JOB_WORKING_DIR)


def test_uv_cache_is_not_on_the_network_backed_cache_volume():
    """A package cache is latency-bound; the volume is for datasets and weights.

    Asserted against the mount path itself, so moving the volume cannot silently
    drag the uv cache onto it.
    """
    assert not UV_CACHE_DIR.startswith(CacheConfig().mount_base_path)


def test_an_explicit_uv_cache_dir_wins():
    """Ours is a default, not an override of what the user asked for."""
    env = _build(
        environment_variables={UV_CACHE_DIR_ENV_VAR: "/somewhere/else"}
    ).job.runtime.environment_variables
    assert env[UV_CACHE_DIR_ENV_VAR] == "/somewhere/else"


def test_build_exec_project_always_mounts_the_cache_volume():
    """Datasets and weights a rerun would otherwise re-download survive on it."""
    runtime = _build().job.runtime
    assert runtime.cache_config is not None
    assert runtime.cache_config.enabled is True
    # Pinning to one node would make a long-running command unschedulable.
    assert runtime.cache_config.require_cache_affinity is False
    # Still no checkpointing: a one-off command has no checkpoints to write.
    assert runtime.checkpointing_config.enabled is False


def test_exec_mounts_the_cache_volume(tmp_path):
    result, mock_push = _invoke_exec(["--"] + USER_COMMAND, tmp_path)

    assert result.exit_code == 0, result.output
    assert mock_push.call_args[1]["config"].job.runtime.cache_config.enabled is True


def test_build_exec_project_with_uv_selects_the_uv_image_on_cpu():
    job = _build(
        start_command=USER_COMMAND, project_name="my-project", project=UvProject()
    ).job
    assert job.image.base_image == UV_BASE_IMAGE
    # The uv image already ships uv, so no install step is prepended.
    assert job.runtime.start_commands == [USER_COMMAND_STR]


def test_build_exec_project_with_uv_installs_uv_on_the_gpu_image():
    job = _build(
        start_command=USER_COMMAND,
        project_name="my-project",
        accelerator="H100",
        project=UvProject(),
    ).job
    assert job.image.base_image == DEFAULT_BASE_IMAGE
    assert "astral.sh/uv/install.sh" in job.runtime.start_commands[0]
    assert job.runtime.start_commands[0].endswith(f"{USER_COMMAND_STR}'")


def test_build_exec_project_with_uv_installs_uv_on_a_custom_image():
    job = _build(
        start_command=USER_COMMAND,
        project_name="my-project",
        base_image="my-registry/my-image:latest",
        project=UvProject(),
    ).job
    assert job.image.base_image == "my-registry/my-image:latest"
    assert "astral.sh/uv/install.sh" in job.runtime.start_commands[0]
    assert job.runtime.start_commands[0].endswith(f"{USER_COMMAND_STR}'")


@pytest.mark.parametrize("accelerator", [None, "H100"])
def test_build_exec_project_without_with_uv_injects_nothing(accelerator):
    job = _build(
        start_command=USER_COMMAND, project_name="my-project", accelerator=accelerator
    ).job
    assert job.runtime.start_commands == [USER_COMMAND_STR]


# --- uv install failure mode, secret/env edge cases, workspace root ----------


def test_uv_install_step_does_not_pipe_curl_into_sh():
    """`curl | sh` would exit 0 on a failed download, hiding the error until the
    job died with an opaque `uv: not found`."""
    command = build_start_commands(
        start_command=["uv", "--version"], setup_steps=UV_INSTALL_STEPS
    )[0]
    script = command[len("/bin/sh -c ") :]
    # The install group must be its own command list, not a pipeline into sh.
    assert "install.sh | sh" not in script
    assert "-o /tmp/uv-install.sh && sh /tmp/uv-install.sh" in script
    assert (
        subprocess.run(
            ["sh", "-n", "-c", script.strip("'")], capture_output=True
        ).returncode
        == 0
    )


def test_uv_install_step_reports_failure_when_every_route_fails():
    """The group must exit non-zero, not fall through to an opaque `uv: not found`."""
    script = (
        UV_INSTALL_STEPS[0]
        .replace(UV_PRESENT_PROBE, "false")
        .replace(UV_PIP_INSTALL, "false")
        .replace(UV_CURL_INSTALL, "false")
    )
    # Every route replaced by identity, so nothing here reaches the network.
    assert "curl" not in script and "pip" not in script
    assert subprocess.run(["sh", "-c", script], capture_output=True).returncode != 0


@pytest.mark.parametrize("entry", ["FOO=", "FOO"])
def test_parse_environment_variables_rejects_an_empty_secret_name(entry):
    with pytest.raises(click.UsageError, match="Invalid --secret value"):
        parse_environment_variables(secrets=[entry])


def test_parse_environment_variables_still_allows_an_empty_env_value():
    """Unlike a secret name, an empty --env value is legitimate."""
    assert parse_environment_variables(env=["EMPTY="]) == {"EMPTY": ""}


def test_is_uv_project_accepts_a_lockfile(tmp_path):
    (tmp_path / UV_LOCK_FILE).write_text("")
    assert is_uv_project(tmp_path)


def test_is_uv_project_accepts_a_tool_uv_section(tmp_path):
    (tmp_path / PYPROJECT_FILE).write_text(
        "[project]\nname = 'x'\n\n[tool.uv]\ndev-dependencies = []\n"
    )
    assert is_uv_project(tmp_path)


def test_is_uv_project_rejects_a_poetry_project(tmp_path):
    """A pyproject.toml alone is not a uv project -- poetry, hatch, PDM and
    setuptools all ship one, and warning on those is noise."""
    (tmp_path / PYPROJECT_FILE).write_text(
        "[tool.poetry]\nname = 'x'\nversion = '0.1.0'\n"
    )
    assert not is_uv_project(tmp_path)


def test_is_uv_project_rejects_malformed_toml(tmp_path):
    (tmp_path / PYPROJECT_FILE).write_text("this is [not valid toml")
    assert not is_uv_project(tmp_path)


def test_is_uv_project_rejects_a_plain_directory(tmp_path):
    assert not is_uv_project(tmp_path)


def test_resolve_workspace_root_defaults_to_the_source_dir(tmp_path):
    assert resolve_workspace_root(tmp_path, None) == tmp_path


def test_resolve_workspace_root_handles_relative_and_absolute(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    assert resolve_workspace_root(child, "..") == tmp_path.resolve()
    assert resolve_workspace_root(child, str(tmp_path)) == tmp_path.resolve()


def test_validate_workspace_root_accepts_a_parent(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    assert validate_workspace_root(child, "..") == tmp_path.resolve()


def test_validate_workspace_root_rejects_a_non_parent(tmp_path):
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    here = tmp_path / "here"
    here.mkdir()
    with pytest.raises(click.UsageError, match="does not contain the current"):
        validate_workspace_root(here, str(sibling))


def test_validate_workspace_root_rejects_a_missing_directory(tmp_path):
    with pytest.raises(click.UsageError, match="not a directory"):
        validate_workspace_root(tmp_path, "nope-does-not-exist")


# --- CLI ---------------------------------------------------------------------


TEAM_ID = "team1"


def _mock_remote(secrets=("my_api_key",), provisioned=True):
    """A remote whose team already holds `secrets`.

    `provisioned` includes the per-team API-key secret, which is the steady state
    after any earlier run, so a test only opts out to exercise first-use minting.
    """
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = Mock(spec=BasetenApi)
    mock_remote.api.get_teams.return_value = {
        "team-a": TeamType(id=TEAM_ID, name="team-a", default=True)
    }
    mock_remote.api.list_training_projects.return_value = []
    names = list(secrets)
    if provisioned:
        names.append(team_api_key_secret_name(TEAM_ID))
    mock_remote.api.get_team_secrets.return_value = {
        "secrets": [{"name": name} for name in names]
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


def _invoke_exec(args, cwd: Path, tail: bool = False, remote=None, runner=None):
    """Invoke `truss loops exec` from `cwd`, returning (result, mock_push)."""
    base_args = ["loops", "exec", "--remote", "test_remote"]
    if tail:
        base_args.append("--tail")
    remote = remote if remote is not None else _mock_remote()

    with (
        _chdir(cwd),
        patch("truss_train.public_api.push") as mock_push,
        patch(
            "truss.cli.loops_commands.RemoteFactory.get_remote_team", return_value=None
        ),
        patch("truss.cli.loops_commands.RemoteFactory.create", return_value=remote),
    ):
        mock_push.return_value = {
            "id": "job123",
            "training_project": {"id": "proj123", "name": cwd.name},
        }
        result = (runner or CliRunner()).invoke(truss_cli, base_args + list(args))

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
    assert job.compute.cpu_count == LOOPS_EXEC_CPU_COUNT
    assert job.compute.memory == LOOPS_EXEC_MEMORY


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
        ["--cpu-count", "8", "--memory", "32Gi", "--", "python", "my_script.py"],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    compute = mock_push.call_args[1]["config"].job.compute
    # Both differ from the defaults, so this cannot pass by accident.
    assert compute.cpu_count == 8
    assert compute.memory == "32Gi"


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
        UV_CACHE_DIR_ENV_VAR: UV_CACHE_DIR,
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


def test_exec_warns_about_a_uv_project_without_with_uv(tmp_path):
    result, _ = _invoke_exec(["--", "python", "my_script.py"], _uv_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" in _message_text(result)


def test_exec_does_not_warn_for_a_poetry_project(tmp_path):
    """The warning is about uv specifically; a bare pyproject.toml is not a signal."""
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'x'\n")

    result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" not in _message_text(result)


def test_exec_does_not_warn_when_with_uv_is_passed(tmp_path):
    result, _ = _invoke_exec(["--with-uv", "--"] + USER_COMMAND, _uv_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" not in _message_text(result)


def test_exec_does_not_warn_for_a_plain_directory(tmp_path):
    result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" not in _message_text(result)


# --- secret existence validation ---------------------------------------------


def _api(secrets_response):
    # spec'd: renaming the listing method should fail loudly, not silently disable
    # this check.
    api = Mock(spec=BasetenApi)
    api.get_team_secrets.return_value = secrets_response
    api.get_all_secrets.return_value = secrets_response
    return api


def _validate(api, environment_variables, team_id=TEAM_ID):
    validate_secret_references(api, environment_variables, team_id=team_id)


def test_validate_secret_references_skips_the_call_without_any_secret_refs():
    api = _api({"secrets": []})

    _validate(api, {"PLAIN": "literal"})

    api.get_team_secrets.assert_not_called()
    api.get_all_secrets.assert_not_called()


def test_validate_secret_references_checks_the_job_team_not_the_whole_workspace():
    """The server resolves a SecretReference against the project's team, so a
    workspace-wide listing would pass names the push then rejects."""
    api = _api({"secrets": [{"name": "my_api_key"}]})

    _validate(api, {"K": SecretReference(name="my_api_key")})

    api.get_team_secrets.assert_called_once_with(TEAM_ID)
    api.get_all_secrets.assert_not_called()


def test_validate_secret_references_falls_back_to_the_workspace_without_a_team():
    """No team means the organization's default team, which the team-scoped endpoint
    cannot address; the caller-wide listing is a superset, so it can only miss a
    failure, never invent one."""
    api = _api({"secrets": [{"name": "my_api_key"}]})

    _validate(api, {"K": SecretReference(name="my_api_key")}, team_id=None)

    api.get_all_secrets.assert_called_once_with()
    api.get_team_secrets.assert_not_called()


def test_validate_secret_references_errors_when_the_secret_is_absent():
    api = _api({"secrets": [{"name": "other"}]})

    with pytest.raises(click.UsageError) as excinfo:
        _validate(api, {"K": SecretReference(name="my_api_key")})

    message = str(excinfo.value)
    assert "Secret my_api_key was not found in this team's secrets" in message
    assert "Create it at" in message
    assert SECRETS_SETTINGS_URL in message


def test_validate_secret_references_names_the_workspace_scope_in_the_error():
    with pytest.raises(click.UsageError) as excinfo:
        _validate(
            _api({"secrets": []}),
            {"K": SecretReference(name="my_api_key")},
            team_id=None,
        )

    assert "this workspace's secrets" in str(excinfo.value)


def test_validate_secret_references_errors_for_an_empty_team():
    """An empty listing is a real answer, unlike an unreadable one."""
    with pytest.raises(click.UsageError, match="my_api_key"):
        _validate(_api({"secrets": []}), {"K": SecretReference(name="my_api_key")})


def test_validate_secret_references_names_every_missing_secret():
    with pytest.raises(click.UsageError) as excinfo:
        _validate(
            _api({"secrets": [{"name": "present"}]}),
            {
                "A": SecretReference(name="missing_a"),
                "B": SecretReference(name="present"),
                "C": SecretReference(name="missing_b"),
                "D": "literal",
            },
        )

    message = str(excinfo.value)
    assert "missing_a" in message and "missing_b" in message
    assert "present" not in message
    assert "Secrets missing_a, missing_b were not found" in message
    assert "Create them at" in message


def test_validate_secret_references_accepts_the_documented_payload_shape():
    _validate(
        _api({"secrets": [{"id": "abc", "name": "my_api_key", "team_name": "t"}]}),
        {"K": SecretReference(name="my_api_key")},
    )


@pytest.mark.parametrize(
    "response",
    [
        "a string",
        42,
        None,
        {"data": []},
        {"secrets": "nope"},
        [123],
        # Shapes the endpoint does not return; treating them as authoritative would
        # report a present secret as missing.
        ["my_api_key"],
        [{"name": "my_api_key"}],
        {"secrets": ["my_api_key"]},
    ],
)
def test_validate_secret_references_stays_silent_on_an_unreadable_payload(response):
    """Rather than guess a shape and fail over a secret that is really there."""
    _validate(_api(response), {"K": SecretReference(name="my_api_key")})


def test_validate_secret_references_swallows_api_errors():
    api = Mock(spec=BasetenApi)
    api.get_team_secrets.side_effect = RuntimeError("403 Forbidden")

    _validate(api, {"K": SecretReference(name="my_api_key")})


def test_validate_secret_references_reports_the_name_verbatim():
    """click.UsageError messages are not markup-parsed, so escaping them would leak
    a backslash into the name the user sees."""
    with pytest.raises(click.UsageError) as excinfo:
        validate_secret_references(
            _api({"secrets": []}),
            {"K": SecretReference(name="[bold]weird")},
            team_id=None,
        )

    message = str(excinfo.value)
    assert "Secret [bold]weird was not found" in message
    assert "\\" not in message


# --- api key provisioning ----------------------------------------------------


def test_team_api_key_secret_name_is_stable_per_team():
    """One name per team, so a second run reuses the first run's key."""
    assert team_api_key_secret_name("team1") == "truss-team-team1-exec-api-key"
    assert team_api_key_secret_name("team2") != team_api_key_secret_name("team1")


def test_ensure_team_api_key_secret_reuses_an_existing_secret():
    name = team_api_key_secret_name(TEAM_ID)
    api = _api({"secrets": [{"name": name}]})

    assert ensure_team_api_key_secret(api, TEAM_ID) == SecretReference(name=name)

    # Nothing minted: the stored key is still valid and its value is unreadable.
    api.create_api_key.assert_not_called()
    api.upsert_team_secret.assert_not_called()


def test_ensure_team_api_key_secret_mints_and_stores_when_absent():
    api = _api({"secrets": [{"name": "unrelated"}]})
    api.create_api_key.return_value = {"api_key": "sk-abc"}
    name = team_api_key_secret_name(TEAM_ID)

    assert ensure_team_api_key_secret(api, TEAM_ID) == SecretReference(name=name)

    api.create_api_key.assert_called_once_with(
        APIKeyCategory.WORKSPACE_MANAGE_ALL, name, team_id=TEAM_ID
    )
    api.upsert_team_secret.assert_called_once_with(TEAM_ID, name, "sk-abc")


@pytest.mark.parametrize(
    "response", [{"unexpected": "shape"}, {"secrets": "nope"}, None]
)
def test_ensure_team_api_key_secret_does_not_mint_on_an_unreadable_listing(response):
    """Minting against a listing we can't read risks replacing a working key."""
    api = _api(response)

    assert ensure_team_api_key_secret(api, TEAM_ID) is None

    api.create_api_key.assert_not_called()
    api.upsert_team_secret.assert_not_called()


def test_ensure_team_api_key_secret_gives_up_when_minting_fails():
    api = _api({"secrets": []})
    api.create_api_key.side_effect = RuntimeError("403 Forbidden")

    assert ensure_team_api_key_secret(api, TEAM_ID) is None

    api.upsert_team_secret.assert_not_called()


def test_ensure_team_api_key_secret_reports_a_response_without_a_key_value():
    """The key may exist server-side even though its value never reached us."""
    api = _api({"secrets": []})
    api.create_api_key.return_value = {"prefix": "sk-abc"}

    with pytest.raises(OrphanedApiKeyError, match=team_api_key_secret_name(TEAM_ID)):
        ensure_team_api_key_secret(api, TEAM_ID)

    # A secret holding no key would fail at runtime, not at push time.
    api.upsert_team_secret.assert_not_called()


def test_a_key_that_cannot_be_stored_never_reaches_the_terminal(tmp_path):
    """The CLI renders frame locals at debug log levels, so a plaintext key held in
    a frame that raises is a key printed to the user's screen."""
    remote = _mock_remote(secrets=(), provisioned=False)
    remote.api.create_api_key.return_value = {"api_key": "sk-must-not-appear"}
    remote.api.upsert_team_secret.side_effect = RuntimeError("503")

    result, mock_push = _invoke_exec(
        ["--log", "DEBUG", "--", "python", "my_script.py"], tmp_path, remote=remote
    )

    assert result.exit_code != 0
    assert "sk-must-not-appear" not in result.output
    assert API_KEYS_SETTINGS_URL in _message_text(result)
    mock_push.assert_not_called()


def test_ensure_team_api_key_secret_reports_a_key_it_could_not_store():
    """The key is live, unreferenced and unrevocable through this client, so
    swallowing this would accumulate credentials silently."""
    api = _api({"secrets": []})
    api.create_api_key.return_value = {"api_key": "sk-abc"}
    api.upsert_team_secret.side_effect = RuntimeError("503")

    with pytest.raises(OrphanedApiKeyError) as excinfo:
        ensure_team_api_key_secret(api, TEAM_ID)

    message = str(excinfo.value)
    assert API_KEYS_SETTINGS_URL in message
    assert "sk-abc" not in message


def test_ensure_team_api_key_secret_swallows_a_failed_listing():
    api = Mock(spec=BasetenApi)
    api.get_team_secrets.side_effect = RuntimeError("403 Forbidden")

    assert ensure_team_api_key_secret(api, TEAM_ID) is None

    api.create_api_key.assert_not_called()


def test_exec_errors_and_does_not_push_when_a_secret_is_missing(tmp_path):
    """Absent from a listing we could read means the job would fail to start."""
    remote = _mock_remote(secrets=("some_other_secret",))

    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=remote,
    )

    assert result.exit_code != 0
    assert "my_api_key was not found in this team" in _message_text(result)
    assert SECRETS_SETTINGS_URL in _message_text(result)
    mock_push.assert_not_called()


def test_exec_pushes_when_the_secret_exists(tmp_path):
    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=_mock_remote(secrets=("my_api_key",)),
    )

    assert result.exit_code == 0, result.output
    assert "was not found in this team" not in _message_text(result)
    mock_push.assert_called_once()


def test_exec_still_pushes_when_listing_secrets_fails(tmp_path):
    remote = _mock_remote()
    remote.api.get_team_secrets.side_effect = RuntimeError("403 Forbidden")

    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=remote,
    )

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert "403 Forbidden" not in result.output
    mock_push.assert_called_once()


def test_exec_still_pushes_when_the_secrets_payload_is_unreadable(tmp_path):
    """An unparseable listing is an API problem, not proof the secret is missing."""
    remote = _mock_remote()
    remote.api.get_team_secrets.return_value = {"unexpected": "shape"}

    result, mock_push = _invoke_exec(
        ["--secret", "BASETEN_API_KEY=my_api_key", "--", "python", "my_script.py"],
        tmp_path,
        remote=remote,
    )

    assert result.exit_code == 0, result.output
    assert "was not found in this team" not in _message_text(result)
    mock_push.assert_called_once()


def test_exec_with_uv_uses_the_uv_image_even_without_uv_metadata(tmp_path):
    """--with-uv is about the image, not about detection: it names uv explicitly, so
    it applies whether or not the directory carries uv metadata."""
    result, mock_push = _invoke_exec(["--with-uv", "--"] + USER_COMMAND, tmp_path)

    assert result.exit_code == 0, result.output
    job = mock_push.call_args[1]["config"].job
    assert job.image.base_image == UV_BASE_IMAGE
    assert job.runtime.start_commands == [USER_COMMAND_STR]


def test_exec_provisions_the_api_key_secret_without_secret_flags(tmp_path):
    """Nothing to validate, but the job still needs a Baseten credential."""
    secret_name = team_api_key_secret_name("team1")
    remote = _mock_remote(secrets=(secret_name,))

    result, mock_push = _invoke_exec(
        ["--env", "PLAIN=1", "--", "python", "my_script.py"], tmp_path, remote=remote
    )

    assert result.exit_code == 0, result.output
    environment_variables = mock_push.call_args[1][
        "config"
    ].job.runtime.environment_variables
    assert environment_variables[BASETEN_API_KEY_ENV_VAR] == SecretReference(
        name=secret_name
    )
    assert environment_variables["PLAIN"] == "1"
    remote.api.create_api_key.assert_not_called()
    assert secret_name in _message_text(result)


def test_exec_does_not_override_an_explicit_baseten_api_key(tmp_path):
    """An explicit --secret is the user's decision and must win."""
    remote = _mock_remote(secrets=("my_own_key",))

    result, mock_push = _invoke_exec(
        [
            "--secret",
            f"{BASETEN_API_KEY_ENV_VAR}=my_own_key",
            "--",
            "python",
            "my_script.py",
        ],
        tmp_path,
        remote=remote,
    )

    assert result.exit_code == 0, result.output
    environment_variables = mock_push.call_args[1][
        "config"
    ].job.runtime.environment_variables
    assert environment_variables[BASETEN_API_KEY_ENV_VAR] == SecretReference(
        name="my_own_key"
    )
    remote.api.create_api_key.assert_not_called()


def test_exec_no_api_key_skips_provisioning_entirely(tmp_path):
    """A job that should carry no credential must not even mint one."""
    remote = _mock_remote(secrets=(), provisioned=False)

    result, mock_push = _invoke_exec(
        ["--no-api-key", "--", "python", "my_script.py"], tmp_path, remote=remote
    )

    assert result.exit_code == 0, result.output
    remote.api.create_api_key.assert_not_called()
    remote.api.upsert_team_secret.assert_not_called()
    environment_variables = mock_push.call_args[1][
        "config"
    ].job.runtime.environment_variables
    assert BASETEN_API_KEY_ENV_VAR not in environment_variables


def test_exec_still_pushes_when_the_api_key_cannot_be_provisioned(tmp_path):
    """A command that never calls the Baseten API needs no credential."""
    remote = _mock_remote(secrets=(), provisioned=False)
    remote.api.create_api_key.side_effect = RuntimeError("403 Forbidden")

    result, mock_push = _invoke_exec(
        ["--", "python", "my_script.py"], tmp_path, remote=remote
    )

    assert result.exit_code == 0, result.output
    message = _message_text(result)
    assert "could not provision a team Baseten API key" in message
    assert "403 Forbidden" not in message
    environment_variables = mock_push.call_args[1][
        "config"
    ].job.runtime.environment_variables
    assert BASETEN_API_KEY_ENV_VAR not in environment_variables


# --- json output -------------------------------------------------------------


def test_exec_json_output_is_the_only_thing_on_stdout(tmp_path):
    """So `truss loops exec -o json | jq` works: progress goes to stderr."""
    result, mock_push = _invoke_exec(
        ["-o", "json", "--"] + USER_COMMAND, tmp_path, runner=_split_streams_runner()
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(_strip_ansi(result.stdout))
    assert payload["job_id"] == "job123"
    assert payload["project"]["id"] == "proj123"
    assert payload["project"]["name"] == tmp_path.name
    assert payload["ssh_hostname"] == "training-job-job123-0.ssh.baseten.co"
    assert payload["start_command"] == USER_COMMAND_STR
    assert payload["compute"] == {
        "cpu_count": LOOPS_EXEC_CPU_COUNT,
        "memory": LOOPS_EXEC_MEMORY,
        "accelerator": None,
        "gpu_count": None,
    }
    # The prose the default format prints is absent from stdout entirely.
    assert "Job created" not in result.stdout
    assert "Launching" not in result.stdout


def test_exec_json_output_reports_the_environment_the_job_gets():
    """Names only -- an --env value can be as sensitive as a secret."""
    result, mock_push = _invoke_exec(
        ["-o", "json", "--env", "TOKEN=hunter2", "--"] + USER_COMMAND,
        Path("/tmp"),
        runner=_split_streams_runner(),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(_strip_ansi(result.stdout))
    assert "TOKEN" in payload["environment_variables"]
    assert UV_CACHE_DIR_ENV_VAR in payload["environment_variables"]
    assert "hunter2" not in result.stdout


def test_exec_default_format_prints_prose_not_json(tmp_path):
    result, _ = _invoke_exec(["--"] + USER_COMMAND, tmp_path)

    assert result.exit_code == 0, result.output
    assert "Job created" in _plain(result.stdout)
    with pytest.raises(json.JSONDecodeError):
        json.loads(_strip_ansi(result.stdout))


def test_exec_does_not_tail_by_default(tmp_path):
    """The motivating use case is a client running for hours, so blocking the
    terminal is the wrong default; matches `push` and `workstation`."""
    with patch("truss.cli.loops_commands.TrainingLogWatcher") as mock_watcher:
        result, mock_push = _invoke_exec(["--", "python", "my_script.py"], tmp_path)

    assert result.exit_code == 0, result.output
    mock_push.assert_called_once()
    mock_watcher.assert_not_called()


def test_exec_tails_when_asked(tmp_path):
    with patch("truss.cli.loops_commands.TrainingLogWatcher") as mock_watcher:
        mock_watcher.return_value.watch.return_value = []
        mock_watcher.return_value.failed = False
        result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path, tail=True)

    assert result.exit_code == 0, result.output
    assert mock_watcher.call_args[0][1:] == ("proj123", "job123")


def test_exec_exits_nonzero_when_the_job_fails(tmp_path):
    """Otherwise `truss loops exec --tail -- pytest` is green in CI regardless of
    outcome. --tail is passed explicitly: it is opt-in, so this cannot rely on a
    default."""
    with patch("truss.cli.loops_commands.TrainingLogWatcher") as mock_watcher:
        mock_watcher.return_value.watch.return_value = []
        mock_watcher.return_value.failed = True
        result, mock_push = _invoke_exec(
            ["--", "python", "my_script.py"], tmp_path, tail=True
        )

    assert result.exit_code == 1, result.output
    mock_push.assert_called_once()
    # A clean exit, not an error surfaced by the common_options handler.
    assert "ERROR" not in _message_text(result)
    assert "Traceback" not in result.output


def test_exec_exits_zero_when_the_job_succeeds(tmp_path):
    with patch("truss.cli.loops_commands.TrainingLogWatcher") as mock_watcher:
        mock_watcher.return_value.watch.return_value = []
        mock_watcher.return_value.failed = False
        result, _ = _invoke_exec(["--", "python", "my_script.py"], tmp_path, tail=True)

    assert result.exit_code == 0, result.output


def test_exec_escapes_brackets_in_the_launch_line(tmp_path):
    """A directory named `myproj[v2]` must not be reported as `myproj`."""
    work_dir = tmp_path / "myproj[v2]"
    work_dir.mkdir()

    result, mock_push = _invoke_exec(["--", "python", "my_script.py"], work_dir)

    assert result.exit_code == 0, result.output
    assert "myproj[v2]" in _message_text(result)
    assert mock_push.call_args[1]["config"].name == "myproj[v2]"


def test_exec_survives_a_closing_tag_in_the_project_name(tmp_path):
    """`[/cyan]` in interpolated text would otherwise raise a rich MarkupError.

    A directory name can't contain `/`, but --project-name can.
    """
    result, mock_push = _invoke_exec(
        ["--project-name", "weird[/cyan]name", "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert "MarkupError" not in result.output
    assert mock_push.call_args[1]["config"].name == "weird[/cyan]name"


@pytest.mark.parametrize("value", ["0", "-4"])
def test_exec_rejects_a_nonpositive_cpu_count(value, tmp_path):
    result, mock_push = _invoke_exec(
        ["--cpu-count", value, "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code != 0
    mock_push.assert_not_called()


def test_exec_rejects_an_empty_secret_name(tmp_path):
    result, mock_push = _invoke_exec(
        ["--secret", "KEY=", "--", "python", "my_script.py"], tmp_path
    )

    assert result.exit_code != 0
    assert "Invalid --secret value" in _message_text(result)
    mock_push.assert_not_called()


def test_exec_rejects_a_workspace_root_that_is_not_a_parent(tmp_path):
    """Client-side, so a bad value can't leave a stray empty training project."""
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    here = tmp_path / "here"
    here.mkdir()

    result, mock_push = _invoke_exec(
        ["--workspace-root", str(sibling), "--", "python", "my_script.py"], here
    )

    assert result.exit_code != 0
    assert "does not contain the current" in _message_text(result)
    mock_push.assert_not_called()


def test_exec_uv_warning_follows_the_workspace_root(tmp_path):
    """With --workspace-root the parent is what gets archived and executed, so that
    is the directory whose uv metadata matters."""
    (tmp_path / "uv.lock").write_text("")
    child = tmp_path / "child"
    child.mkdir()

    result, _ = _invoke_exec(
        ["--workspace-root", "..", "--", "python", "my_script.py"], child
    )

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" in _message_text(result)


def test_exec_uv_warning_ignores_the_cwd_when_workspace_root_is_set(tmp_path):
    """The mirror of the above: uv metadata in the cwd is irrelevant when the
    archived root is elsewhere."""
    child = tmp_path / "child"
    child.mkdir()
    (child / "uv.lock").write_text("")

    result, _ = _invoke_exec(
        ["--workspace-root", "..", "--", "python", "my_script.py"], child
    )

    assert result.exit_code == 0, result.output
    assert "--with-uv was not passed" not in _message_text(result)


def test_exec_no_tail_flag_is_still_accepted(tmp_path):
    """Now the same as the default, but the paired form must keep working."""
    with patch("truss.cli.loops_commands.TrainingLogWatcher") as mock_watcher:
        result, _ = _invoke_exec(
            ["--no-tail", "--", "python", "my_script.py"], tmp_path
        )

    assert result.exit_code == 0, result.output
    mock_watcher.assert_not_called()
