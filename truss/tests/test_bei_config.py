import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml
from click.testing import CliRunner

from truss.base.trt_llm_config import ImageVersions
from truss.base.truss_config import TrussConfig
from truss.contexts import docker_build_setup as setup
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)


@pytest.fixture
def bei_dir(test_data_path, tmp_path):
    return shutil.copytree(test_data_path / "test_bei_truss", tmp_path / "truss")


@pytest.fixture
def config_data(bei_dir):
    return yaml.safe_load((bei_dir / "config.yaml").read_text())


def _argv(config):
    return shlex.split(config.docker_server.start_command.splitlines()[-1])[1:]


def test_no_build_configuration_and_roundtrip(config_data, bei_dir, tmp_path):
    config = TrussConfig.from_dict(config_data)
    config.validate_forbid_extra()
    assert config.docker_server.no_build is True
    assert config.docker_server.server_port == 8080
    assert config.environment_variables["BEI_PORT"] == "8080"
    assert "--port=8080" in _argv(config)
    assert config.docker_server.predict_endpoint == "/v1/embeddings"
    assert config.docker_server.readiness_endpoint == "/health"
    assert TrussConfig.from_dict(config.to_dict(verbose=False)) == config
    assert _argv(config).count("--backend-device-ids=auto") == 2
    ServingImageBuilderContext.run(bei_dir).prepare_image_build_dir(tmp_path / "build")
    dockerfile = (tmp_path / "build/Dockerfile").read_text()
    assert "FROM example.com/bei:test-nvidia" in dockerfile
    assert not any(
        line.startswith(("RUN ", "COPY ")) for line in dockerfile.splitlines()
    )
    assert not (tmp_path / "build/supervisord.conf").exists()


@pytest.mark.parametrize("mode", ["none", "fp8", "mxfp8", "nvfp4"])
def test_quantization_maps_to_current_native_flags(config_data, mode):
    config_data["bei"]["configuration"][0]["quantization"] = mode
    args = _argv(TrussConfig.from_dict(config_data))
    assert f"--dynamic-quantization={mode}" in args
    dtypes = [arg for arg in args if arg.startswith("--dtype=")]
    assert dtypes[0] == (
        "--dtype=bfloat16" if mode in ("mxfp8", "nvfp4") else "--dtype=auto"
    )


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "duplicate",
        "delimiter",
        "unknown_quantization",
        "tpu_multi",
        "tpu_quantization",
    ],
)
def test_invalid_model_configuration(config_data, case):
    models = config_data["bei"]["configuration"]
    if case == "empty":
        models.clear()
    elif case == "duplicate":
        models[1]["served_model_name"] = models[0]["served_model_name"]
    elif case == "delimiter":
        models[0]["revision"] = "main;other"
    elif case == "unknown_quantization":
        models[0]["quantization"] = "unknown"
    else:
        config_data["bei"]["backend"] = "tpu"
        if case == "tpu_quantization":
            config_data["bei"]["configuration"] = [models[1]]
    with pytest.raises(ValueError):
        TrussConfig.from_dict(config_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("requirements", ["numpy"]),
        ("system_packages", ["curl"]),
        ("build_commands", ["echo build"]),
    ],
)
def test_no_build_rejects_install_settings(config_data, field, value):
    config_data[field] = value
    with pytest.raises(ValueError, match="no_build"):
        TrussConfig.from_dict(config_data)


def test_tpu_native_defaults(config_data):
    config_data["bei"] = {
        "backend": "tpu",
        "configuration": [{"checkpoint": "Alibaba-NLP/gte-multilingual-base"}],
    }
    config = TrussConfig.from_dict(config_data)
    assert "--backend=python-backend" in _argv(config)
    assert "--dtype=bfloat16" in _argv(config)
    assert config.environment_variables["BEI_MAX_BATCH_REQUESTS"] == "32"


def test_shared_image_injection_is_authoritative(
    config_data, bei_dir, tmp_path, monkeypatch
):
    config_data["bei"]["version_override"] = "v3-test"
    (bei_dir / "config.yaml").write_text(yaml.safe_dump(config_data))
    image = "private.example/bei:resolved@sha256:" + "a" * 64
    versions = ImageVersions(
        bei_image="legacy/bei",
        beibert_image="legacy/bert",
        briton_image="legacy/briton",
        v2_llm_image="legacy/v2",
        bei_v3_image=image,
    )
    monkeypatch.delenv("PATCHES_DIR", raising=False)
    for name, path in {
        "TRUSS_SRC_DIR": bei_dir,
        "TRUSS_HASH_FILE": tmp_path / "hash",
        "TRUSS_SIGNATURE_FILE": tmp_path / "signature",
        "TRUSS_BUILD_CONTEXT_DIR": tmp_path / "build",
    }.items():
        monkeypatch.setattr(setup, name, path)
    result = CliRunner().invoke(
        setup.docker_build_setup,
        [
            "--truss_type",
            "server",
            "--trt_llm_image_versions_json",
            versions.model_dump_json(),
        ],
    )
    assert result.exit_code == 0, result.output
    config = TrussConfig.from_yaml(tmp_path / "build/config.yaml")
    assert config.base_image.image == image
    assert config.bei.version_override == "v3-test"
    assert (
        ImageVersions.model_validate(
            {
                key: value
                for key, value in versions.model_dump().items()
                if key != "bei_v3_image"
            }
        ).bei_v3_image
        is None
    )


def test_native_shell_preserves_arguments_and_reads_runtime_secret(
    config_data, tmp_path
):
    checkpoint = r'''/models/space $(touch pwned) `id` 'quote' "double"'''
    config_data["bei"]["configuration"] = [{"checkpoint": checkpoint}]
    config_data["secrets"] = ["hf_access_token"]
    config = TrussConfig.from_dict(config_data)
    fake = tmp_path / "bei.py"
    fake.write_text(
        "import json, os, sys\nprint(json.dumps([sys.argv[1:], os.environ['BEI_HF_TOKEN']]))\n"
    )
    command = config.docker_server.start_command.replace(
        "exec /usr/local/bin/bei ",
        "exec " + shlex.join([sys.executable, str(fake)]) + " ",
    )
    token_file = tmp_path / "token"
    token_file.write_text("test-token")
    command = command.replace("/secrets/hf_access_token", shlex.quote(str(token_file)))
    result = subprocess.run(
        ["sh", "-c", command],
        cwd=tmp_path,
        env={**os.environ, **config.environment_variables},
        capture_output=True,
        text=True,
        check=True,
    )
    args, token = json.loads(result.stdout)
    assert f"--model-id={checkpoint}" in args
    assert token == "test-token"
    assert "test-token" not in command
    assert not (tmp_path / "pwned").exists()


def test_managed_configuration_refreshes_and_can_be_removed(config_data):
    config = TrussConfig.from_dict(config_data)
    config.bei.configuration[0].served_model_name = "updated"
    data = config.to_dict()
    assert "--served-model-name=updated" in data["docker_server"]["start_command"]
    data.pop("bei")
    assert TrussConfig.from_dict(data).docker_server is None


@pytest.mark.parametrize("accelerator", ["TPU_V6E:8", "TPU_V7:4"])
def test_managed_tpu_image_can_be_resolved_after_validation(accelerator):
    config = TrussConfig.from_dict(
        {
            "resources": {"accelerator": accelerator},
            "bei": {
                "backend": "tpu",
                "configuration": [{"checkpoint": "BAAI/bge-small-en-v1.5"}],
            },
        }
    )
    assert config.base_image is None
    assert config.docker_server.no_build is True
