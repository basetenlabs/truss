import copy
import json
import os
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
        ({"requirements": ["numpy"]}, "does not install Python requirements"),
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
    argv = json.loads((build_dir / "bei/config.json").read_text())["argv"]
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
    assert [arg for arg in argv if arg.startswith("--backend-device-ids=")] == [
        "--backend-device-ids=auto",
        "--backend-device-ids=auto",
    ]
    assert "--torch-device=cuda" in argv
    dockerfile = (build_dir / "Dockerfile").read_text()
    assert "FROM example.com/bei:test-nvidia" in dockerfile
    assert "./bei ${APP_HOME}/bei" in dockerfile
    assert "CMD []" in dockerfile
    assert "truss-transfer-cli" not in dockerfile
    assert "/v1/embeddings" in (build_dir / "proxy.conf").read_text()
    assert "/app/bei/launch.py" in (build_dir / "supervisord.conf").read_text()
    # Reloading the generated config and preparing a second context must work.
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
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(bei_config))
    builder = ServingImageBuilderContext.run(bei_truss_dir)
    builder.prepare_image_build_dir(tmp_path / "build")
    runtime = json.loads((tmp_path / "build/bei/config.json").read_text())
    assert "--backend=python-backend" in runtime["argv"]
    assert "--dtype=bfloat16" in runtime["argv"]
    assert not any(
        arg.startswith("--python-config") or arg.startswith("--torch-device")
        for arg in runtime["argv"]
    )
    assert runtime["environment_defaults"] == {"BEI_MAX_BATCH_REQUESTS": "32"}


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
    ["TRUSS_SECRET_hf_access_token", "HF_TOKEN", "BEI_HF_TOKEN", "default"],
)
def test_launcher_passes_literal_arguments_and_runtime_secret(
    bei_truss_dir, bei_config, tmp_path, secret_source
):
    # A local model path may contain shell metacharacters; they must remain data.
    local_path = "/models/model with spaces $(touch pwned) `id` %percent"
    bei_config["bei"]["configuration"] = [
        {"checkpoint": local_path, "served_model_name": "encoder"}
    ]
    bei_config["secrets"] = (
        {"hf_access_token": "runtime-test-token"}
        if secret_source == "default"
        else ["hf_access_token"]
    )
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(bei_config))
    builder = ServingImageBuilderContext.run(bei_truss_dir)
    builder.prepare_image_build_dir(tmp_path / "build")
    runtime_path = tmp_path / "build/bei/config.json"
    runtime = json.loads(runtime_path.read_text())
    fake_server = tmp_path / "fake_bei.py"
    fake_server.write_text(
        "import json, os, sys\nprint(json.dumps({'argv': sys.argv[1:], 'token': os.environ.get('BEI_HF_TOKEN')}))\n"
    )
    runtime["argv"][:1] = [sys.executable, str(fake_server)]
    runtime_path.write_text(json.dumps(runtime))
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("TRUSS_SECRET_hf_access_token", "HF_TOKEN", "BEI_HF_TOKEN")
    }
    if secret_source != "default":
        environment[secret_source] = "runtime-test-token"
    result = subprocess.run(
        [sys.executable, str(tmp_path / "build/bei/launch.py")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    captured = json.loads(result.stdout)
    assert f"--model-id={local_path}" in captured["argv"]
    assert captured["token"] == "runtime-test-token"
    assert not (tmp_path / "pwned").exists()
    if secret_source != "default":
        assert "runtime-test-token" not in runtime_path.read_text()


def test_missing_declared_secret_fails_at_runtime(
    bei_truss_dir, bei_config, tmp_path, monkeypatch
):
    bei_config["secrets"] = ["hf_access_token"]
    (bei_truss_dir / "config.yaml").write_text(yaml.safe_dump(bei_config))
    ServingImageBuilderContext.run(bei_truss_dir).prepare_image_build_dir(
        tmp_path / "build"
    )
    environment = copy.copy(os.environ)
    for key in ("BEI_HF_TOKEN", "HF_TOKEN", "TRUSS_SECRET_hf_access_token"):
        environment.pop(key, None)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "build/bei/launch.py")],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode != 0
    assert "hf_access_token secret is missing" in result.stderr


@pytest.mark.parametrize("slim", [False, True])
def test_python_free_base_image_has_managed_launcher(
    bei_truss_dir, tmp_path, monkeypatch, slim
):
    monkeypatch.setenv("BT_USE_DOCKER_SERVER_SLIM", str(slim).lower())
    ServingImageBuilderContext.run(bei_truss_dir).prepare_image_build_dir(
        tmp_path / "build"
    )
    dockerfile = (tmp_path / "build/Dockerfile").read_text()
    config = TrussConfig.from_yaml(tmp_path / "build/config.yaml")
    assert config.docker_server.start_command.startswith(
        "/docker_server/.venv/bin/python "
    )
    assert "uv venv /docker_server/.venv" in dockerfile
    assert "Supplied base image does not have" not in dockerfile
    assert "ENV BEI_PORT=7997" in dockerfile
    if slim:
        assert 'ENTRYPOINT ["sh", "-c"]' in dockerfile
        assert "./supervisord.conf" not in dockerfile
    else:
        assert "CMD []" in dockerfile


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
