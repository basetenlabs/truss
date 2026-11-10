import json
import os
import shlex
import shutil
import subprocess
import sys
from unittest.mock import patch

import pydantic
import pytest
import yaml
from click.testing import CliRunner

from truss.base.bei_config import BEIImageVersions
from truss.base.truss_config import TrussConfig
from truss.contexts import docker_build_setup as setup
from truss.contexts.docker_build_setup import _fill_bei_versions
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
from truss.truss_handle.truss_handle import TrussHandle


def _native_argv(config):
    return shlex.split(config.docker_server.start_command.splitlines()[-1])[1:]


def _run_native_command(config, tmp_path, environment=None, command=None):
    fake_server = tmp_path / "fake_bei.py"
    fake_server.write_text(
        "import json, os, sys\nprint(json.dumps({'argv': sys.argv[1:], 'token': os.environ.get('BEI_HF_TOKEN')}))\n"
    )
    command = command or config.docker_server.start_command
    command = command.replace(
        "exec /usr/local/bin/bei ",
        "exec " + shlex.join([sys.executable, str(fake_server)]) + " ",
    )
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "BEI_HF_TOKEN",
            "HF_TOKEN",
            "TRUSS_SECRET_hf_access_token",
            "TRUSS_BEI_HF_TOKEN_DEFAULT",
        )
    }
    return subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **clean_environment,
            **config.environment_variables,
            **(environment or {}),
        },
    )


@pytest.fixture
def bei_truss_dir(test_data_path, tmp_path):
    return shutil.copytree(test_data_path / "test_bei_truss", tmp_path / "truss")


@pytest.fixture
def bei_config(bei_truss_dir):
    return yaml.safe_load((bei_truss_dir / "config.yaml").read_text())


def test_top_level_roundtrip_and_extra_fields(bei_config):
    config = TrussConfig.model_validate(bei_config)
    config.validate_forbid_extra()
    assert TrussConfig.from_dict(config.to_dict(verbose=False)) == config
    assert TrussConfig.from_dict(config.to_dict(verbose=True)) == config
    bei_config["bei"]["configuration"][0]["typo"] = True
    with pytest.raises(ValueError, match="Extra fields not allowed"):
        TrussConfig.model_validate(bei_config).validate_forbid_extra()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"live_reload": True}, "live_reload"),
        ({"resources": {}}, "requires accelerator resources"),
        ({"requirements": ["numpy"]}, "no_build"),
        ({"system_packages": ["curl"]}, "no_build"),
        ({"build_commands": ["echo build"]}, "no_build"),
        ({"environment_variables": {"BEI_PORT": "9000"}}, "manages BEI_PORT"),
        ({"resources": {"instance_type": "V100:4x16"}}, "SM80"),
        ({"resources": {"instance_type": "A100:4x16"}}, "SM89"),
        ({"resources": {"instance_type": "TPU_V6E:8x32"}}, "accelerator family"),
        ({"resources": {"accelerator": "L4", "node_count": 2}}, "single-node"),
        ({"runtime": {"transport": {"kind": "grpc"}}}, "HTTP transport"),
        ({"resources": {"accelerator": "V100"}}, "SM80"),
        ({"resources": {"accelerator": "A100"}}, "SM89"),
        (
            {
                "bei": {
                    "backend": "tpu",
                    "configuration": [{"checkpoint": "BAAI/model"}],
                }
            },
            "accelerator family",
        ),
        (
            {
                "docker_server": {
                    "start_command": "other",
                    "server_port": 9000,
                    "predict_endpoint": "/predict",
                    "readiness_endpoint": "/health",
                    "liveness_endpoint": "/health",
                }
            },
            "manages docker_server",
        ),
    ],
)
def test_rejects_conflicting_server_settings(bei_config, overrides, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        TrussConfig.model_validate({**bei_config, **overrides})


def test_bei_and_trt_llm_cannot_be_combined(bei_config, trtllm_config):
    with pytest.raises(pydantic.ValidationError, match="cannot be combined"):
        TrussConfig.model_validate({**trtllm_config, "bei": bei_config["bei"]})


@pytest.mark.parametrize("device_count", [1, 2])
def test_build_context_without_model_code(bei_truss_dir, tmp_path, device_count):
    source = yaml.safe_load((bei_truss_dir / "config.yaml").read_text())
    source["resources"]["accelerator"] = f"L4:{device_count}"
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(source))
    # The platform must see no_build before it decides whether to build an image.
    source_config = TrussConfig.from_dict(source)
    assert source_config.docker_server.no_build is True
    assert source_config.to_dict()["docker_server"]["no_build"] is True
    builder = ServingImageBuilderContext.run(bei_truss_dir)
    build_dir = tmp_path / "build"
    with patch.object(
        builder, "prepare_trtllm_bei_encoder_build_dir"
    ) as legacy_builder:
        builder.prepare_image_build_dir(build_dir)
        legacy_builder.assert_not_called()
    config = TrussConfig.from_yaml(build_dir / "config.yaml")
    assert config.docker_server.server_port == 7997
    assert config.docker_server.predict_endpoint == "/v1/embeddings"
    assert config.trt_llm is None
    assert not (build_dir / "model").exists()
    argv = _native_argv(config)
    assert [arg for arg in argv if arg.startswith("--model-id=")] == [
        "--model-id=BAAI/bge-small-en-v1.5",
        "--model-id=BAAI/bge-large-en-v1.5",
    ]
    assert [arg for arg in argv if arg.startswith("--revision=")] == [
        "--revision=main",
        "--revision=pinned-large",
    ]
    assert [arg for arg in argv if arg.startswith("--enable-fp8-dynamic=")] == [
        "--enable-fp8-dynamic=false",
        "--enable-fp8-dynamic=true",
    ]
    assert argv.count("--backend-device-ids=auto") == 2
    assert "--torch-device=cuda" in argv
    dockerfile = (build_dir / "Dockerfile").read_text()
    assert "FROM example.com/bei:test-nvidia" in dockerfile
    assert 'ENTRYPOINT ["sh", "-c"]' in dockerfile
    command_json = next(
        line.removeprefix("CMD ")
        for line in dockerfile.splitlines()
        if line.startswith("CMD ")
    )
    assert json.loads(command_json) == [config.docker_server.start_command]
    assert not any(
        line.startswith(("RUN ", "COPY ")) for line in dockerfile.splitlines()
    )
    for path in (
        "bei",
        "proxy.conf",
        "supervisord.conf",
        "docker_server_requirements.txt",
    ):
        assert not (build_dir / path).exists()
    ServingImageBuilderContext.run(build_dir).prepare_image_build_dir(
        tmp_path / "rebuild"
    )


def test_tpu_preserves_plugin_and_uses_bfloat16(bei_truss_dir, bei_config, tmp_path):
    bei_config["bei"] = {
        "backend": "tpu",
        "configuration": [{"checkpoint": "Alibaba-NLP/gte-multilingual-base"}],
    }
    bei_config["resources"] = {"instance_type": "TPU_V6E:8x32"}
    bei_config["base_image"]["image"] = "example.com/bei:test-tpu"
    config = TrussConfig.model_validate(bei_config)
    argv = _native_argv(config)
    assert config.docker_server.no_build is True
    assert "--backend=python-backend" in argv
    assert "--dtype=bfloat16" in argv
    assert not any(
        arg.startswith("--python-config") or arg.startswith("--torch-device")
        for arg in argv
    )
    assert config.environment_variables["BEI_MAX_BATCH_REQUESTS"] == "32"
    assert config.environment_variables["BEI_PORT"] == "7997"


def test_unresolved_image_fails_before_default_cuda_image(
    bei_truss_dir, bei_config, tmp_path
):
    bei_config.pop("base_image")
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(bei_config))
    with pytest.raises(ValueError, match="image is unresolved"):
        ServingImageBuilderContext.run(bei_truss_dir).prepare_image_build_dir(
            tmp_path / "build"
        )


@pytest.mark.parametrize("backend", ["nvidia", "tpu"])
def test_image_injection_and_version_override(
    bei_truss_dir, bei_config, backend, tmp_path
):
    bei_config["bei"]["backend"] = backend
    bei_config["bei"]["configuration"] = [{"checkpoint": "BAAI/model"}]
    bei_config["resources"] = {
        "instance_type": "TPU_V6E:8x32" if backend == "tpu" else "L4:4x16"
    }
    bei_config.pop("base_image")
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(bei_config))
    handle = TrussHandle(bei_truss_dir)
    versions = BEIImageVersions(
        nvidia_image="registry/bei:pinned-nvidia", tpu_image="registry/bei:pinned-tpu"
    )
    _fill_bei_versions(handle, versions)
    assert handle.spec.config.base_image.image == f"registry/bei:pinned-{backend}"
    assert "bei" in yaml.safe_load((bei_truss_dir / "config.yaml").read_text())
    handle.spec.config.bei.version_override = "v3.0.2-tested"
    handle.spec.config.write_to_yaml_file(bei_truss_dir / "config.yaml", verbose=True)
    builder = ServingImageBuilderContext.run(bei_truss_dir)
    builder.prepare_image_build_dir(tmp_path / "build")
    assert (
        f"FROM baseten/bei:v3.0.2-tested-{backend}"
        in (tmp_path / "build/Dockerfile").read_text()
    )


def test_explicit_image_is_preserved_by_default_injection(bei_truss_dir):
    handle = TrussHandle(bei_truss_dir)
    _fill_bei_versions(handle, BEIImageVersions(nvidia_image="registry/bei:default"))
    assert handle.spec.config.base_image.image == "example.com/bei:test-nvidia"


@pytest.mark.parametrize(
    "secret_source",
    ["TRUSS_SECRET_hf_access_token", "HF_TOKEN", "BEI_HF_TOKEN", "default", "mounted"],
)
def test_native_command_passes_literal_arguments_and_runtime_secret(
    bei_config, tmp_path, secret_source
):
    local_path = r"""/models/model with spaces $(touch pwned) `id` %percent 'quote' "double" \\escape"""
    bei_config["bei"]["configuration"] = [
        {"checkpoint": local_path, "served_model_name": "encoder"}
    ]
    bei_config["secrets"] = (
        {"hf_access_token": "runtime-test-token"}
        if secret_source == "default"
        else ["hf_access_token"]
    )
    config = TrussConfig.model_validate(bei_config)
    environment = {}
    command = config.docker_server.start_command
    if secret_source == "mounted":
        secret_path = tmp_path / "token"
        secret_path.write_text("runtime-test-token")
        command = command.replace(
            "/secrets/hf_access_token", shlex.quote(str(secret_path))
        )
        environment["TRUSS_SECRET_hf_access_token"] = "overridden-token"
    elif secret_source != "default":
        environment[secret_source] = "runtime-test-token"
    result = _run_native_command(config, tmp_path, environment, command=command)
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert f"--model-id={local_path}" in captured["argv"]
    assert captured["token"] == "runtime-test-token"
    assert not (tmp_path / "pwned").exists()
    assert "runtime-test-token" not in command


def test_missing_declared_secret_fails_at_runtime(bei_config, tmp_path):
    bei_config["secrets"] = ["hf_access_token"]
    config = TrussConfig.model_validate(bei_config)
    result = _run_native_command(config, tmp_path)
    assert result.returncode != 0
    assert "hf_access_token secret is missing" in result.stderr


@pytest.mark.parametrize("slim", [False, True])
def test_no_build_has_no_added_runtime_dependencies(
    bei_truss_dir, tmp_path, monkeypatch, slim
):
    monkeypatch.setenv("BT_USE_DOCKER_SERVER_SLIM", str(slim).lower())
    ServingImageBuilderContext.run(bei_truss_dir).prepare_image_build_dir(
        tmp_path / "build"
    )
    dockerfile = (tmp_path / "build/Dockerfile").read_text()
    assert "python" not in dockerfile.lower()
    assert "nginx" not in dockerfile.lower()
    assert "supervisord" not in dockerfile.lower()
    assert "uv " not in dockerfile
    assert not any(line.startswith("RUN ") for line in dockerfile.splitlines())


def test_remote_build_setup_accepts_bei_image_injection(
    bei_truss_dir, bei_config, tmp_path, monkeypatch
):
    bei_config.pop("base_image")
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(bei_config))
    monkeypatch.delenv("PATCHES_DIR", raising=False)
    monkeypatch.setattr(setup, "TRUSS_SRC_DIR", bei_truss_dir)
    monkeypatch.setattr(setup, "TRUSS_HASH_FILE", tmp_path / "hash")
    monkeypatch.setattr(setup, "TRUSS_SIGNATURE_FILE", tmp_path / "signature")
    monkeypatch.setattr(setup, "TRUSS_BUILD_CONTEXT_DIR", tmp_path / "build")
    result = CliRunner().invoke(
        setup.docker_build_setup,
        [
            "--truss_type",
            "server",
            "--bei_image_versions_json",
            json.dumps({"nvidia_image": "registry/bei:approved-nvidia"}),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "FROM registry/bei:approved-nvidia"
        in (tmp_path / "build/Dockerfile").read_text()
    )
    assert (tmp_path / "hash").is_file()


def test_managed_no_build_settings_refresh_after_config_edits(bei_config):
    config = TrussConfig.model_validate(bei_config)
    assert config.docker_server.no_build is True
    config.bei.configuration[0].served_model_name = "updated-name"
    changed = config.model_copy(update={"secrets": ["hf_access_token"]})
    exported = changed.to_dict()
    assert (
        "--served-model-name=updated-name" in exported["docker_server"]["start_command"]
    )
    assert "/secrets/hf_access_token" in exported["docker_server"]["start_command"]
    assert exported["docker_server"]["no_build"] is True
    assert TrussConfig.from_dict(exported).to_dict() == exported


def test_version_override_resolves_before_any_build(bei_config):
    bei_config.pop("base_image")
    bei_config["bei"]["version_override"] = "v3.0.2-tested"
    config = TrussConfig.model_validate(bei_config)
    assert config.base_image.image == "baseten/bei:v3.0.2-tested-nvidia"
    assert config.docker_server.no_build is True
    assert "/app/" not in config.docker_server.start_command


@pytest.mark.parametrize("method", ["assignment", "copy", "serialized"])
def test_removing_bei_clears_managed_deployment_settings(bei_config, method):
    bei_config["bei"]["version_override"] = "v3.0.2-tested"
    config = TrussConfig.model_validate(bei_config)
    if method == "assignment":
        config.bei = None
    elif method == "copy":
        config = config.model_copy(update={"bei": None})
    else:
        data = config.to_dict()
        data.pop("bei")
        config = TrussConfig.from_dict(data)
    data = config.to_dict()
    assert data["docker_server"] is None
    assert data["base_image"] is None
    assert "BEI_PORT" not in data["environment_variables"]
    assert "BEI_HOSTNAME" not in data["environment_variables"]


def test_removing_version_override_allows_default_image_resolution(
    bei_config, bei_truss_dir
):
    bei_config.pop("base_image")
    bei_config["bei"]["version_override"] = "v3.0.2-tested"
    config = TrussConfig.model_validate(bei_config)
    config.bei.version_override = None
    data = config.to_dict()
    assert data["base_image"]["image"] == ""
    config = TrussConfig.from_dict(data)
    config.write_to_yaml_file(bei_truss_dir / "config.yaml")
    handle = TrussHandle(bei_truss_dir)
    _fill_bei_versions(
        handle, BEIImageVersions(nvidia_image="registry/bei:approved-nvidia")
    )
    assert handle.spec.config.base_image.image == "registry/bei:approved-nvidia"


@pytest.mark.parametrize(
    "empty_source", ["mounted", "TRUSS_SECRET_hf_access_token", "default"]
)
def test_empty_secret_source_preserves_hf_token_fallback(
    bei_config, tmp_path, empty_source
):
    bei_config["secrets"] = (
        {"hf_access_token": ""} if empty_source == "default" else ["hf_access_token"]
    )
    config = TrussConfig.model_validate(bei_config)
    command = config.docker_server.start_command
    environment = {"HF_TOKEN": "fallback-test-token"}
    if empty_source == "mounted":
        token_file = tmp_path / "empty-token"
        token_file.write_text("")
        command = command.replace(
            "/secrets/hf_access_token", shlex.quote(str(token_file))
        )
    elif empty_source == "TRUSS_SECRET_hf_access_token":
        environment[empty_source] = ""
    result = _run_native_command(config, tmp_path, environment, command=command)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["token"] == "fallback-test-token"
