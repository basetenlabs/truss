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
from truss.base.truss_config import Resources, TrussConfig
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


def test_checkpoints_generate_runtime_weights(config_data):
    config_data["secrets"] = ["hf_access_token"]
    config = TrussConfig.from_dict(config_data)
    assert [source.source for source in config.weights.sources] == [
        "hf://BAAI/bge-small-en-v1.5@main",
        "hf://BAAI/bge-large-en-v1.5@pinned-large",
    ]
    for model, source in zip(config.bei.configuration, config.weights.sources):
        assert f"--model-id={source.mount_location}" in _argv(config)
        assert f"--served-model-name={model.served_model_name}" in _argv(config)
        assert source.ignore_patterns == ["*.bin", "onnx/**"]
        assert source.auth_secret_name == "hf_access_token"
    assert config.bei.configuration[0].checkpoint == "BAAI/bge-small-en-v1.5"
    assert TrussConfig.from_dict(config.to_dict(verbose=False)) == config


def test_explicit_hf_weights_override_defaults_and_are_shared(config_data):
    config_data["bei"]["configuration"][1]["checkpoint"] = "BAAI/bge-small-en-v1.5"
    config_data["bei"]["configuration"][1].pop("revision")
    explicit = {
        "source": "hf://baai/bge-small-en-v1.5",
        "mount_location": "/models/bge-small",
        "auth_secret_name": "custom_hf_token",
        "allow_patterns": ["*.json", "*.safetensors"],
        "ignore_patterns": ["onnx/**"],
    }
    config_data["weights"] = [explicit]
    config = TrussConfig.from_dict(config_data)
    assert config.to_dict(verbose=False)["weights"] == [explicit]
    assert _argv(config).count("--model-id=/models/bge-small") == 2


def test_automatic_weights_share_download_but_not_revisions(config_data):
    models = config_data["bei"]["configuration"]
    models[1]["checkpoint"] = models[0]["checkpoint"]
    config = TrussConfig.from_dict(config_data)
    assert len(config.weights.sources) == 2
    assert (
        config.weights.sources[0].mount_location
        != config.weights.sources[1].mount_location
    )
    models[1].pop("revision")
    config = TrussConfig.from_dict(config_data)
    assert len(config.weights.sources) == 1
    assert (
        _argv(config).count("--model-id=" + config.weights.sources[0].mount_location)
        == 2
    )
    assert config.weights.sources[0].auth_secret_name is None


def test_absolute_checkpoints_use_manual_bucket_mounts(config_data):
    config_data["bei"]["configuration"] = [{"checkpoint": "/models/custom/bge-small"}]
    config_data["weights"] = [
        {"source": "s3://my-models/encoders", "mount_location": "/models/custom"}
    ]
    config = TrussConfig.from_dict(config_data)
    assert config.to_dict(verbose=False)["weights"] == config_data["weights"]
    assert "--model-id=/models/custom/bge-small" in _argv(config)


def test_automatic_mounts_do_not_overwrite_explicit_weights(config_data):
    config_data["weights"] = [
        {"source": "gs://my-models/encoders", "mount_location": "/models/bei-0"}
    ]
    config = TrussConfig.from_dict(config_data)
    assert [source.mount_location for source in config.weights.sources] == [
        "/models/bei-0",
        "/models/bei-1",
        "/models/bei-2",
    ]


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


@pytest.mark.parametrize(
    "field", ["restart_threshold_seconds", "stop_traffic_threshold_seconds"]
)
@pytest.mark.parametrize("value", [None, 0, -1, 61])
def test_health_thresholds_must_be_explicit_and_tight(config_data, field, value):
    checks = config_data["runtime"]["health_checks"]
    if value is None:
        checks.pop(field)
    else:
        checks[field] = value
    with pytest.raises(ValueError, match=field):
        TrussConfig.from_dict(config_data)


def test_health_checks_cannot_use_platform_defaults(config_data):
    config_data.pop("runtime")
    with pytest.raises(ValueError, match="runtime.health_checks"):
        TrussConfig.from_dict(config_data)


def test_health_limits_revalidated_after_nested_edits(config_data):
    config = TrussConfig.from_dict(config_data)
    config.runtime.health_checks.restart_threshold_seconds = 61
    with pytest.raises(ValueError, match="restart_threshold_seconds"):
        config.to_dict()


@pytest.mark.parametrize("value", [1, 60])
def test_health_threshold_boundaries_allow_long_startup(config_data, value):
    config_data["runtime"]["health_checks"] = {
        "restart_threshold_seconds": value,
        "stop_traffic_threshold_seconds": value,
        "startup_threshold_seconds": 1800,
        "restart_check_delay_seconds": 900,
    }
    config = TrussConfig.from_dict(config_data)
    assert config.runtime.health_checks.startup_threshold_seconds == 1800
    assert config.runtime.health_checks.restart_check_delay_seconds == 900


def test_tpu_native_defaults(config_data):
    config_data["bei"] = {
        "backend": "tpu",
        "configuration": [{"checkpoint": "Alibaba-NLP/gte-multilingual-base"}],
    }
    config = TrussConfig.from_dict(config_data)
    assert "--backend=python-backend" in _argv(config)
    assert "--dtype=bfloat16" in _argv(config)
    assert config.environment_variables["BEI_MAX_BATCH_REQUESTS"] == "32"


@pytest.mark.parametrize("inject_image", [False, True])
@pytest.mark.parametrize("explicit_image", [False, True])
def test_shared_image_injection_supports_truss_first_rollout(
    config_data, bei_dir, tmp_path, monkeypatch, inject_image, explicit_image
):
    config_data["bei"]["version_override"] = "v3-test"
    if not explicit_image:
        config_data.pop("base_image")
    (bei_dir / "config.yaml").write_text(yaml.safe_dump(config_data))
    image = "private.example/bei:resolved@sha256:" + "a" * 64
    versions = ImageVersions(
        bei_image="legacy/bei",
        beibert_image="legacy/bert",
        briton_image="legacy/briton",
        v2_llm_image="legacy/v2",
        bei_v3_image=image if inject_image else None,
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
            versions.model_dump_json(exclude_none=True),
        ],
    )
    if not inject_image and not explicit_image:
        assert isinstance(result.exception, ValueError)
        assert "BEIv3 image is unresolved" in str(result.exception)
        return
    assert result.exit_code == 0, result.output
    config = TrussConfig.from_yaml(tmp_path / "build/config.yaml")
    assert config.base_image.image == (
        image if inject_image else config_data["base_image"]["image"]
    )
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


def test_managed_tpu_image_can_be_resolved_after_validation(config_data, monkeypatch):
    monkeypatch.setattr(Resources, "is_tpu", property(lambda self: True))
    config = TrussConfig.from_dict(
        {
            "runtime": config_data["runtime"],
            "bei": {
                "backend": "tpu",
                "configuration": [{"checkpoint": "BAAI/bge-small-en-v1.5"}],
            },
        }
    )
    assert config.base_image is None
    assert config.docker_server.no_build is True
