import tempfile
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import packaging.version
import pydantic
import pytest
import yaml

from truss.base import constants
from truss.base.trt_llm_config import TrussTRTLLMQuantizationType
from truss.base.truss_config import (
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    Accelerator,
    AcceleratorSpec,
    BaseImage,
    Build,
    CacheInternal,
    CheckpointList,
    DockerAuthSettings,
    DockerAuthType,
    DockerServer,
    HTTPOptions,
    ModelCache,
    ModelRepo,
    ModelRepoCacheInternal,
    Resources,
    Runtime,
    TransportKind,
    TrussConfig,
    WebsocketOptions,
    Weights,
    WeightsSource,
    _map_to_supported_python_version,
)
from truss.truss_handle.truss_handle import TrussHandle


@pytest.mark.parametrize(
    "input_dict, expect_resources, output_dict",
    [
        (
            {},
            Resources(),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": False,
                "accelerator": None,
            },
        ),
        (
            {"accelerator": None},
            Resources(),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": False,
                "accelerator": None,
            },
        ),
        (
            {"accelerator": "V100"},
            Resources(
                accelerator=AcceleratorSpec(accelerator=Accelerator.V100, count=1)
            ),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": True,
                "accelerator": "V100",
            },
        ),
        (
            {"accelerator": "T4:1"},
            Resources(accelerator=AcceleratorSpec(accelerator=Accelerator.T4, count=1)),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": True,
                "accelerator": "T4",
            },
        ),
        (
            {"accelerator": "A10G:4"},
            Resources(
                accelerator=AcceleratorSpec(accelerator=Accelerator.A10G, count=4)
            ),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": True,
                "accelerator": "A10G:4",
            },
        ),
        (
            {"node_count": 2},
            Resources(node_count=2),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": False,
                "accelerator": None,
                "node_count": 2,
            },
        ),
    ],
)
def test_parse_resources(input_dict, expect_resources, output_dict):
    parsed_result = Resources.model_validate(input_dict)
    assert parsed_result == expect_resources
    assert parsed_result.to_dict(verbose=True) == output_dict


@pytest.mark.parametrize(
    "input_dict, expect_resources, output_dict",
    [
        (
            {"instance_type": "L4:8x32"},
            Resources(instance_type="L4:8x32"),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": False,
                "accelerator": None,
                "instance_type": "L4:8x32",
            },
        ),
        (
            {"instance_type": "H100:8x80"},
            Resources(instance_type="H100:8x80"),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": False,
                "accelerator": None,
                "instance_type": "H100:8x80",
            },
        ),
        (
            {"instance_type": "CPU:4x16"},
            Resources(instance_type="CPU:4x16"),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": False,
                "accelerator": None,
                "instance_type": "CPU:4x16",
            },
        ),
    ],
)
def test_parse_resources_with_instance_type(input_dict, expect_resources, output_dict):
    parsed_result = Resources.model_validate(input_dict)
    assert parsed_result == expect_resources
    assert parsed_result.to_dict(verbose=True) == output_dict


def test_instance_type_not_serialized_when_none():
    """Test that instance_type is omitted from serialization when not set."""
    resources = Resources()
    result = resources.to_dict(verbose=True)
    assert "instance_type" not in result


@pytest.mark.parametrize(
    "cpu_spec, expected_valid",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("1.5", True),
        ("1.5m", True),
        (1, False),
        ("1m", True),
        ("1M", False),
        ("M", False),
        ("M1", False),
    ],
)
def test_validate_cpu_spec(cpu_spec, expected_valid):
    if not expected_valid:
        with pytest.raises(pydantic.ValidationError):
            Resources(cpu=cpu_spec)
    else:
        Resources(cpu=cpu_spec)


@pytest.mark.parametrize(
    "mem_spec, expected_valid, memory_in_bytes",
    [
        (None, False, None),
        (1, False, None),
        ("1m", False, None),
        ("1k", True, 10**3),
        ("512k", True, 512 * 10**3),
        ("512M", True, 512 * 10**6),
        ("1.5Gi", True, 1.5 * 1024**3),
        ("abc", False, None),
        ("1024", True, 1024),
    ],
)
def test_validate_mem_spec(mem_spec, expected_valid, memory_in_bytes):
    if not expected_valid:
        with pytest.raises(pydantic.ValidationError):
            Resources(memory=mem_spec)
    else:
        assert memory_in_bytes == Resources(memory=mem_spec).memory_in_bytes


@pytest.mark.parametrize(
    "input_str, expected_acc",
    [
        # ("", AcceleratorSpec(accelerator=None, count=0)),
        ("T4", AcceleratorSpec(accelerator=Accelerator.T4, count=1)),
        ("A10G:4", AcceleratorSpec(accelerator=Accelerator.A10G, count=4)),
        ("A100:8", AcceleratorSpec(accelerator=Accelerator.A100, count=8)),
        ("A100_40GB", AcceleratorSpec(accelerator=Accelerator.A100_40GB, count=1)),
        ("H100", AcceleratorSpec(accelerator=Accelerator.H100, count=1)),
        ("H200", AcceleratorSpec(accelerator=Accelerator.H200, count=1)),
        ("H100_40GB", AcceleratorSpec(accelerator=Accelerator.H100_40GB, count=1)),
        ("B200", AcceleratorSpec(accelerator=Accelerator.B200, count=1)),
    ],
)
def test_acc_spec_from_str(input_str, expected_acc):
    assert AcceleratorSpec.model_validate(input_str) == expected_acc


@pytest.mark.parametrize(
    "input_dict, expect_base_image, output_dict",
    [
        (
            {},
            BaseImage(),
            {"image": "", "python_executable_path": "", "docker_auth": None},
        ),
        (
            {"image": "custom_base_image", "python_executable_path": "/path/python"},
            BaseImage(image="custom_base_image", python_executable_path="/path/python"),
            {
                "image": "custom_base_image",
                "python_executable_path": "/path/python",
                "docker_auth": None,
            },
        ),
        (
            {
                "image": "custom_base_image",
                "python_executable_path": "/path/python",
                "docker_auth": {
                    "auth_method": "GCP_SERVICE_ACCOUNT_JSON",
                    "secret_name": "some-secret-name",
                    "registry": "some-docker-registry",
                },
            },
            BaseImage(
                image="custom_base_image",
                python_executable_path="/path/python",
                docker_auth=DockerAuthSettings(
                    auth_method=DockerAuthType.GCP_SERVICE_ACCOUNT_JSON,
                    secret_name="some-secret-name",
                    registry="some-docker-registry",
                ),
            ),
            {
                "image": "custom_base_image",
                "python_executable_path": "/path/python",
                "docker_auth": {
                    "auth_method": "GCP_SERVICE_ACCOUNT_JSON",
                    "secret_name": "some-secret-name",
                    "registry": "some-docker-registry",
                    "aws_access_key_id_secret_name": "aws_access_key_id",
                    "aws_secret_access_key_secret_name": ("aws_secret_access_key"),
                },
            },
        ),
        (
            {
                "image": "custom_base_image",
                "python_executable_path": "/path/python",
                "docker_auth": {
                    "auth_method": "AWS_IAM",
                    "registry": "some-ecr-docker-registry",
                },
            },
            BaseImage(
                image="custom_base_image",
                python_executable_path="/path/python",
                docker_auth=DockerAuthSettings(
                    auth_method=DockerAuthType.AWS_IAM,
                    registry="some-ecr-docker-registry",
                ),
            ),
            {
                "image": "custom_base_image",
                "python_executable_path": "/path/python",
                "docker_auth": {
                    "auth_method": "AWS_IAM",
                    "registry": "some-ecr-docker-registry",
                    "secret_name": None,
                    "aws_access_key_id_secret_name": "aws_access_key_id",
                    "aws_secret_access_key_secret_name": "aws_secret_access_key",
                },
            },
        ),
    ],
)
def test_parse_base_image(input_dict, expect_base_image, output_dict):
    parsed_result = BaseImage.model_validate(input_dict)
    assert parsed_result == expect_base_image
    assert parsed_result.to_dict(verbose=True) == output_dict


def test_default_config_not_crowded_end_to_end():
    config = TrussConfig(python_version="py39", requirements=[])

    config_yaml = """
python_version: py39
resources:
  accelerator: null
  cpu: '1'
  memory: 2Gi
  use_gpu: false
"""

    assert config_yaml.strip() == yaml.dump(config.to_dict(verbose=False)).strip()


def test_null_cache_internal_key():
    config_yaml_dict = {"cache_internal": None}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        yaml.safe_dump(config_yaml_dict, tmp_file)
    config = TrussConfig.from_yaml(Path(tmp_file.name))
    assert config.cache_internal.models == []


def test_empty_model_cache_key():
    config_yaml_dict = {"model_cache": []}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        yaml.safe_dump(config_yaml_dict, tmp_file)
    config = TrussConfig.from_yaml(Path(tmp_file.name))
    assert config.model_cache.models == []


def test_cache_internal_with_models(default_config):
    config = TrussConfig(
        python_version="py39",
        cache_internal=CacheInternal(
            [
                ModelRepoCacheInternal(repo_id="test/model"),
                ModelRepoCacheInternal(repo_id="test/model2"),
            ]
        ),
    )
    new_config = default_config
    new_config["cache_internal"] = [
        {"repo_id": "test/model"},
        {"repo_id": "test/model2"},
    ]
    assert new_config == config.to_dict(verbose=False)


def test_huggingface_cache_single_model_default_revision(default_config):
    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache([ModelRepo(repo_id="test/model", use_volume=False)]),
    )

    new_config = default_config
    new_config["model_cache"] = [{"repo_id": "test/model", "use_volume": False}]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["model_cache"][0].get("revision") == ""


def test_huggingface_cache_single_model_non_default_revision_v1():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        model_cache=ModelCache(
            [ModelRepo(repo_id="test/model", revision="not-main", use_volume=False)]
        ),
    )

    assert config.to_dict(verbose=False)["model_cache"][0].get("revision") == "not-main"


def test_huggingface_cache_multiple_models_default_revision(default_config):
    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(
            [
                ModelRepo(repo_id="test/model1", revision="main", use_volume=False),
                ModelRepo(repo_id="test/model2", use_volume=False),
            ]
        ),
    )

    new_config = default_config
    new_config["model_cache"] = [
        {"repo_id": "test/model1", "revision": "main", "use_volume": False},
        {"repo_id": "test/model2", "use_volume": False},
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["model_cache"], config.to_dict(verbose=True)[
        "model_cache"
    ]
    assert config.to_dict(verbose=True)["model_cache"][0].get("revision") == "main"
    assert config.to_dict(verbose=True)["model_cache"][1].get("revision") == ""


def test_huggingface_cache_multiple_models_mixed_revision(default_config):
    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(
            [
                ModelRepo(repo_id="test/model1", use_volume=False),
                ModelRepo(
                    repo_id="test/model2", revision="not-main2", use_volume=False
                ),
            ]
        ),
    )

    new_config = default_config
    new_config["model_cache"] = [
        {"repo_id": "test/model1", "use_volume": False},
        {"repo_id": "test/model2", "revision": "not-main2", "use_volume": False},
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["model_cache"][0].get("revision") == ""
    assert config.to_dict(verbose=True)["model_cache"][1].get("revision") == "not-main2"


def test_huggingface_cache_v2_use_volume(default_config):
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        model_cache=ModelCache(
            [
                dict(
                    repo_id="test/model1",
                    revision="main",
                    use_volume=True,
                    volume_folder="test_model1",
                )
            ]
        ),
    )

    new_config = default_config
    new_config["model_cache"] = [
        {
            "repo_id": "test/model1",
            "revision": "main",
            "volume_folder": "test_model1",
            "use_volume": True,
        }
    ]

    assert new_config == config.to_dict(verbose=False)


def test_empty_config(default_config):
    config = TrussConfig()
    new_config = default_config

    assert new_config == config.to_dict(verbose=False)


def test_from_yaml():
    data = {"description": "this is a test"}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        result = TrussConfig.from_yaml(yaml_path)

        assert result.description == "this is a test"


def test_from_yaml_empty():
    data = {}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        result = TrussConfig.from_yaml(yaml_path)

        # test some attributes (should be default)
        assert result.description is None
        assert result.spec_version == "2.0"
        assert result.bundled_packages_dir == "packages"


def test_from_yaml_duplicate_keys():
    yaml_content = """
description: first description
model_name: test-model
description: second description
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    with pytest.warns(UserWarning, match="Detected duplicate key `description`"):
        config = TrussConfig.from_yaml(yaml_path)
    assert config.description == "second description"


def test_from_yaml_duplicate_nested_keys():
    yaml_content = """
resources:
  cpu: "1"
  memory: "2Gi"
  cpu: "2"
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    with pytest.warns(UserWarning, match="Detected duplicate key `cpu`"):
        config = TrussConfig.from_yaml(yaml_path)
    assert config.resources.cpu == "2"


def test_from_yaml_same_key_at_different_nesting_levels():
    yaml_content = """
model_name: test-model
resources:
  cpu: "1"
  memory: "2Gi"
build:
  model_name: build-model-name
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    config = TrussConfig.from_yaml(yaml_path)
    assert config.model_name == "test-model"


def test_from_yaml_secrets_as_list():
    data = {"description": "this is a test", "secrets": ["foo", "bar"]}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        with pytest.raises(ValueError):
            TrussConfig.from_yaml(yaml_path)


def test_from_yaml_python_version():
    invalid_py_version_data = {
        "description": "this is a test",
        "python_version": "py37",
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(invalid_py_version_data, yaml_file)

        with pytest.raises(ValueError):
            TrussConfig.from_yaml(yaml_path)

    valid_py_version_data = {"description": "this is a test", "python_version": "py39"}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(valid_py_version_data, yaml_file)

        result = TrussConfig.from_yaml(yaml_path)
        assert result.python_version == "py39"


def test_from_yaml_environment_variables():
    data = {
        "description": "this is a test",
        "environment_variables": {"foo": "bar", "bool": True, "int": 0},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        result = TrussConfig.from_yaml(yaml_path)
        assert result.environment_variables == {
            "foo": "bar",
            "bool": "true",
            "int": "0",
        }


def test_secret_to_path_mapping_correct_type(default_config):
    data = {
        "description": "this is a test",
        "build": {"secret_to_path_mapping": {"foo": "/bar"}},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        truss_config = TrussConfig.from_yaml(yaml_path)
        assert truss_config.build.secret_to_path_mapping == {"foo": "/bar"}


@pytest.mark.parametrize(
    "secret_name, should_error",
    [
        (None, True),
        (1, True),
        ("", True),
        (".", True),
        ("..", True),
        ("a" * 253, False),
        ("a" * 254, True),
        ("-", False),
        ("-.", False),
        ("a-.", False),
        ("-.a", False),
        ("a-foo", False),
        ("a.foo", False),
        (".foo", False),
        ("x\\", True),
        ("a_b", False),
        ("_a", False),
        ("a_", False),
        ("sd#^Y5^%", True),
    ],
)
def test_validate_secret_name(secret_name, should_error):
    does_error = False
    try:
        Build.validate_secret_name(secret_name)
    except:  # noqa
        does_error = True

    assert does_error == should_error


def test_secret_to_path_mapping_invalid_secret_name(default_config):
    data = {
        "description": "this is a test",
        "build": {"secret_to_path_mapping": {"!foo_bar": "/bar"}},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        with pytest.raises(ValueError):
            TrussConfig.from_yaml(yaml_path)


def test_secret_to_path_mapping_incorrect_type(default_config):
    data = {
        "description": "this is a test",
        "build": {"secret_to_path_mapping": ["something else"]},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(data, yaml_file)

        with pytest.raises(ValueError):
            TrussConfig.from_yaml(yaml_path)


def test_max_beam_width_check(trtllm_config):
    trtllm_config["trt_llm"]["build"]["max_beam_width"] = 2
    with pytest.raises(ValueError):
        TrussConfig.model_validate(trtllm_config)


def test_plugin_paged_context_fmha_check(trtllm_config):
    trtllm_config["trt_llm"]["build"]["plugin_configuration"] = {
        "paged_kv_cache": False,
        "use_paged_context_fmha": True,
        "use_fp8_context_fmha": False,
    }
    with pytest.raises(ValueError):
        TrussConfig.model_validate(trtllm_config)


@pytest.mark.parametrize(
    "repo",
    [
        "./llama-3.1-8b",
        "../my-model-is-in-parent-directory",
        "~/.huggingface/my--model--cache/model",
        "foo.git",
        "datasets/foo/bar",
        ".repo_idother..repo..id",
    ],
)
def test_invalid_hf_repo(trtllm_config, repo):
    trtllm_config["trt_llm"]["build"]["checkpoint_repository"]["source"] = "HF"
    trtllm_config["trt_llm"]["build"]["checkpoint_repository"]["repo"] = repo
    with pytest.raises(ValueError):
        TrussConfig.model_validate(trtllm_config)


def test_plugin_paged_fp8_context_fmha_check(trtllm_config):
    trtllm_config["trt_llm"]["build"]["plugin_configuration"] = {
        "paged_kv_cache": False,
        "use_paged_context_fmha": False,
        "use_fp8_context_fmha": True,
    }
    with pytest.raises(ValueError):
        TrussConfig.model_validate(trtllm_config)
    trtllm_config["trt_llm"]["build"]["plugin_configuration"] = {
        "paged_kv_cache": True,
        "use_paged_context_fmha": False,
        "use_fp8_context_fmha": True,
    }
    with pytest.raises(ValueError):
        TrussConfig.model_validate(trtllm_config)


def test_fp8_context_fmha_check_kv_dtype(trtllm_config):
    trtllm_config["trt_llm"]["build"]["plugin_configuration"] = {
        "paged_kv_cache": True,
        "use_paged_context_fmha": True,
        "use_fp8_context_fmha": True,
    }
    trtllm_config["trt_llm"]["build"]["quantization_type"] = (
        TrussTRTLLMQuantizationType.FP8_KV.value
    )
    TrussConfig.model_validate(trtllm_config)


@pytest.mark.parametrize("verbose, expect_equal", [(False, True), (True, False)])
def test_to_dict_trtllm(verbose, expect_equal, trtllm_config):
    assert (
        TrussConfig.model_validate(trtllm_config).to_dict(verbose=verbose)
        == trtllm_config
    ) == expect_equal


@pytest.mark.parametrize("verbose, expect_equal", [(False, True), (True, False)])
def test_to_dict_trtllm_spec_dec(
    verbose, expect_equal, trtllm_spec_dec_config_lookahead_v1
):
    assert (
        TrussConfig.model_validate(trtllm_spec_dec_config_lookahead_v1).to_dict(
            verbose=verbose
        )
        == trtllm_spec_dec_config_lookahead_v1
    ) == expect_equal


def test_from_yaml_invalid_requirements_configuration():
    invalid_requirements = {
        "requirements_file": "requirements.txt",
        "requirements": ["requests"],
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as yaml_file:
        yaml_path = Path(yaml_file.name)
        yaml.safe_dump(invalid_requirements, yaml_file)

        with pytest.raises(ValueError):
            TrussConfig.from_yaml(yaml_path)


@pytest.mark.parametrize(
    "quant_format, accelerator, expectation",
    [
        (TrussTRTLLMQuantizationType.NO_QUANT, Accelerator.A100, does_not_raise()),
        (TrussTRTLLMQuantizationType.FP8, Accelerator.H100, does_not_raise()),
        (TrussTRTLLMQuantizationType.FP8_KV, Accelerator.H100_40GB, does_not_raise()),
        (
            TrussTRTLLMQuantizationType.NO_QUANT,
            Accelerator.T4,
            pytest.raises(pydantic.ValidationError),
        ),
        (
            TrussTRTLLMQuantizationType.NO_QUANT,
            Accelerator.V100,
            pytest.raises(pydantic.ValidationError),
        ),
        (
            TrussTRTLLMQuantizationType.FP8,
            Accelerator.A100,
            pytest.raises(pydantic.ValidationError),
        ),
        (
            TrussTRTLLMQuantizationType.FP8_KV,
            Accelerator.A100,
            pytest.raises(pydantic.ValidationError),
        ),
    ],
)
def test_validate_quant_format_and_accelerator_for_trt_llm_builder(
    quant_format, accelerator, expectation, custom_model_trt_llm
):
    config = TrussHandle(custom_model_trt_llm).spec.config
    config.trt_llm.build.quantization_type = quant_format
    config.resources.accelerator.accelerator = accelerator
    with expectation:
        TrussConfig.model_validate(config.to_dict())


def test_resources_transport_read_from_new_yaml(tmp_path):
    yaml_content = """
    runtime:
      transport:
        kind: websocket
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = TrussConfig.from_yaml(config_path)
    assert isinstance(config.runtime.transport, WebsocketOptions)
    assert config.runtime.transport.kind == TransportKind.WEBSOCKET
    assert config.runtime.is_websocket_endpoint is True

    config_dict = config.to_dict(verbose=False)
    config_new = TrussConfig.from_dict(config_dict)
    assert config_new.runtime.transport.kind == TransportKind.WEBSOCKET
    assert config_new.runtime.is_websocket_endpoint is True


def test_websocket_options_ping_values(tmp_path):
    yaml_content = """
    runtime:
      transport:
        kind: websocket
        ping_interval_seconds: 5
        ping_timeout_seconds: 10
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = TrussConfig.from_yaml(config_path)
    assert isinstance(config.runtime.transport, WebsocketOptions)
    assert config.runtime.transport.ping_interval_seconds == 5
    assert config.runtime.transport.ping_timeout_seconds == 10

    out_path = tmp_path / "out.yaml"
    config.write_to_yaml_file(out_path, verbose=False)

    dumped = yaml.safe_load(out_path.read_text())
    assert dumped["runtime"]["transport"]["ping_interval_seconds"] == 5
    assert dumped["runtime"]["transport"]["ping_timeout_seconds"] == 10


def test_resources_transport_read_unspecified(tmp_path):
    yaml_content = """
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = TrussConfig.from_yaml(config_path)
    assert isinstance(config.runtime.transport, HTTPOptions)
    assert config.runtime.transport.kind == TransportKind.HTTP
    assert config.runtime.is_websocket_endpoint is False

    config_dict = config.to_dict(verbose=False)
    config_new = TrussConfig.from_dict(config_dict)
    assert config_new.runtime.transport.kind == TransportKind.HTTP
    assert config_new.runtime.is_websocket_endpoint is False


def test_resources_transport_read_empty(tmp_path):
    yaml_content = """
    runtime:
        transport: {}
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = TrussConfig.from_yaml(config_path)
    assert isinstance(config.runtime.transport, HTTPOptions)
    assert config.runtime.transport.kind == TransportKind.HTTP
    assert config.runtime.is_websocket_endpoint is False

    config_dict = config.to_dict(verbose=False)
    config_new = TrussConfig.from_dict(config_dict)
    assert config_new.runtime.transport.kind == TransportKind.HTTP
    assert config_new.runtime.is_websocket_endpoint is False


def test_resources_transport_read_from_legacy_yaml(tmp_path):
    yaml_content = """
    runtime:
      is_websocket_endpoint: true
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    with pytest.warns(DeprecationWarning, match="is_websocket_endpoint"):
        config = TrussConfig.from_yaml(config_path)
    assert isinstance(config.runtime.transport, WebsocketOptions)
    assert config.runtime.transport.kind == TransportKind.WEBSOCKET
    assert config.runtime.is_websocket_endpoint is True


def test_resources_transport_serialize_from_new_way(tmp_path):
    config = TrussConfig(runtime=Runtime(transport=WebsocketOptions()))
    out_path = tmp_path / "out.yaml"
    config.write_to_yaml_file(out_path, verbose=False)

    dumped = yaml.safe_load(out_path.read_text())
    assert dumped["runtime"]["transport"]["kind"] == "websocket"
    assert dumped["runtime"]["is_websocket_endpoint"] is True


def test_resources_transport_serialize_from_old_way(tmp_path):
    yaml_content = """
    runtime:
      is_websocket_endpoint: true
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = TrussConfig.from_yaml(config_path)

    out_path = tmp_path / "out.yaml"
    config.write_to_yaml_file(out_path, verbose=False)

    dumped = yaml.safe_load(out_path.read_text())
    assert dumped["runtime"]["transport"]["kind"] == "websocket"
    assert dumped["runtime"]["is_websocket_endpoint"] is True


def test_resources_transport_correct_serialize_from_legacy(tmp_path):
    config = TrussConfig()

    out_path = tmp_path / "out.yaml"
    config.write_to_yaml_file(out_path, verbose=False)

    dumped = yaml.safe_load(out_path.read_text())
    assert "runtime" not in dumped

    config_new = TrussConfig.from_yaml(out_path)
    assert config_new.runtime.transport.kind == TransportKind.HTTP
    assert config_new.runtime.is_websocket_endpoint is False


def test_resources_transport_correct_serialize(tmp_path):
    config = TrussConfig()
    config.runtime.is_websocket_endpoint = True

    out_path = tmp_path / "out.yaml"
    config.write_to_yaml_file(out_path, verbose=False)

    dumped = yaml.safe_load(out_path.read_text())
    assert dumped["runtime"]["transport"]["kind"] == "websocket"
    assert dumped["runtime"]["is_websocket_endpoint"] is True


def test_resources_transport_correct_serialize_from(tmp_path):
    config = TrussConfig()
    config.runtime.transport = WebsocketOptions()

    out_path = tmp_path / "out.yaml"
    config.write_to_yaml_file(out_path, verbose=False)

    dumped = yaml.safe_load(out_path.read_text())
    assert dumped["runtime"]["transport"]["kind"] == "websocket"
    assert dumped["runtime"]["is_websocket_endpoint"] is True


def test_validate_on_assignment():
    config = TrussConfig()
    with pytest.raises(pydantic.ValidationError, match="should be a valid boolean"):
        config.runtime.is_websocket_endpoint = 123

    with pytest.raises(pydantic.ValidationError):
        config.python_version = "3.7"

    with pytest.raises(pydantic.ValidationError):
        config.python_version = "py27"

    with pytest.raises(pydantic.ValidationError):
        config.python_version = "py27"

    with pytest.raises(
        pydantic.ValidationError, match="should be greater than or equal to 1"
    ):
        config.resources.node_count = -10


def test_validate_extra_fields(tmp_path):
    yaml_with_extra_content = """
    model_name: My Model
    what_is_this_field: true
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_with_extra_content)

    # Plain parsing should pass.
    config = TrussConfig.from_yaml(config_path)
    # Explicit validation with extras not.
    with pytest.raises(
        pydantic.ValidationError,
        match="Extra fields not allowed: \[what_is_this_field\]",
    ):
        config.validate_forbid_extra()


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [
        ("py39", "py39"),
        ("py310", "py310"),
        ("py311", "py311"),
        ("py312", "py312"),
        ("py313", "py313"),
        ("py314", "py314"),
    ],
)
def test_map_to_supported_python_version(python_version, expected_python_version):
    out_python_version = _map_to_supported_python_version(python_version)
    assert out_python_version == expected_python_version


def test_not_supported_python_minor_versions():
    with pytest.raises(
        ValueError,
        match="Mapping python version 3.6 to 3.9, "
        "the lowest version that Truss currently supports.",
    ):
        _map_to_supported_python_version("py36")
    with pytest.raises(
        ValueError,
        match="Mapping python version 3.7 to 3.9, "
        "the lowest version that Truss currently supports.",
    ):
        _map_to_supported_python_version("py37")


def test_not_supported_python_major_versions():
    with pytest.raises(NotImplementedError, match="Only python version 3 is supported"):
        _map_to_supported_python_version("py211")


def test_supported_versions_are_sorted():
    semvers = [packaging.version.parse(v) for v in constants.SUPPORTED_PYTHON_VERSIONS]
    semvers_sorted = sorted(semvers)
    assert semvers == semvers_sorted, (
        f"{constants.SUPPORTED_PYTHON_VERSIONS} must be sorted ascendingly"
    )


def test_clear_runtime_fields():
    config = TrussConfig(
        python_version="py39",
        training_checkpoints=CheckpointList(
            download_folder="/tmp", checkpoints=[], artifact_references=[]
        ),
        environment_variables={"FOO": "BAR"},
        weights=Weights(
            [
                WeightsSource(
                    source="hf://meta-llama/Llama-3.1-8B@main",
                    mount_location="/app/weights",
                )
            ]
        ),
    )

    config.clear_runtime_fields()
    assert config.python_version == "py39"
    assert config.training_checkpoints is None
    assert config.environment_variables == {}
    assert config.weights == Weights([])


def test_docker_server_start_command_single_line_valid():
    """Single-line start_command should be valid."""
    docker_server = DockerServer(
        start_command='sh -c "vllm serve model --port 8000"',
        server_port=8000,
        predict_endpoint="/v1/chat/completions",
        readiness_endpoint="/health",
        liveness_endpoint="/health",
    )
    assert docker_server.start_command == 'sh -c "vllm serve model --port 8000"'


def test_docker_server_start_command_with_newline_valid():
    """start_command containing newlines should be valid (handled by configparser)."""
    multiline_command = "sh -c '\necho hello\n'"
    docker_server = DockerServer(
        start_command=multiline_command,
        server_port=8000,
        predict_endpoint="/v1/chat/completions",
        readiness_endpoint="/health",
        liveness_endpoint="/health",
    )
    assert docker_server.start_command == multiline_command


@pytest.mark.parametrize("yaml_file", ["literal_block.yaml", "folded_block.yaml"])
def test_docker_server_start_command_yaml_with_newlines_valid(
    test_data_path, yaml_file
):
    """YAML syntaxes that preserve/add newlines (| and >) are now valid."""
    config_path = test_data_path / "docker_server_start_command" / yaml_file

    config = TrussConfig.from_yaml(config_path)
    # These YAML syntaxes preserve newlines, which is now supported
    assert "\n" in config.docker_server.start_command


@pytest.mark.parametrize(
    "yaml_file, expected_command",
    [
        ("folded_chomped.yaml", "sh -c /app/server"),
        ("plain_multiline.yaml", "sh -c /app/server"),
        ("backslash_continuation.yaml", "sh -c \\ /app/minimal-server"),
    ],
)
def test_docker_server_start_command_yaml_folding(
    test_data_path, yaml_file, expected_command
):
    """YAML syntaxes like >- and plain multiline fold newlines to spaces."""
    config_path = test_data_path / "docker_server_start_command" / yaml_file

    config = TrussConfig.from_yaml(config_path)
    assert "\n" not in config.docker_server.start_command
    assert config.docker_server.start_command == expected_command


@pytest.mark.parametrize(
    "run_as_user_id,expected,raises",
    [
        pytest.param(1000, 1000, does_not_raise(), id="valid_nonzero"),
        pytest.param(None, None, does_not_raise(), id="default_none"),
        pytest.param(
            0,
            None,
            pytest.raises(pydantic.ValidationError, match="run_as_user_id cannot be 0"),
            id="zero_rejected",
        ),
    ],
)
def test_docker_server_run_as_user_id(run_as_user_id, expected, raises):
    with raises:
        docker_server = DockerServer(
            start_command="python main.py",
            server_port=8000,
            predict_endpoint="/predict",
            readiness_endpoint="/health",
            liveness_endpoint="/health",
            run_as_user_id=run_as_user_id,
        )
        assert docker_server.run_as_user_id == expected


# =============================================================================
# Weights Configuration Tests
# =============================================================================


class TestWeightsSource:
    """Tests for the new WeightsSource model."""

    def test_huggingface_source_basic(self):
        """HuggingFace source with revision in URI should work."""
        source = WeightsSource(
            source="hf://meta-llama/Llama-2-7b@main", mount_location="/models/llama"
        )
        assert source.source == "hf://meta-llama/Llama-2-7b@main"
        assert source.mount_location == "/models/llama"
        assert source.is_huggingface is True
        assert source.auth_secret_name is None

    def test_huggingface_source_with_patterns(self):
        """HuggingFace source with allow/ignore patterns."""
        source = WeightsSource(
            source="hf://meta-llama/Llama-2-7b@main",
            mount_location="/models/llama",
            allow_patterns=["*.safetensors", "config.json"],
            ignore_patterns=["*.md"],
        )
        assert source.allow_patterns == ["*.safetensors", "config.json"]
        assert source.ignore_patterns == ["*.md"]

    def test_s3_source_basic(self):
        """S3 source should work."""
        source = WeightsSource(
            source="s3://my-bucket/models/llama",
            mount_location="/models/llama",
            auth_secret_name="aws_credentials",
        )
        assert source.source == "s3://my-bucket/models/llama"
        assert source.is_huggingface is False

    def test_gcs_source_basic(self):
        """GCS source should work without revision."""
        source = WeightsSource(
            source="gs://my-bucket/models/llama",
            mount_location="/models/llama",
            auth_secret_name="gcp_service_account",
        )
        assert source.source == "gs://my-bucket/models/llama"
        assert source.is_huggingface is False

    def test_azure_source_basic(self):
        """Azure source should work without revision."""
        source = WeightsSource(
            source="azure://myaccount/container/llama",
            mount_location="/models/llama",
            auth_secret_name="azure_credentials",
        )
        assert source.source == "azure://myaccount/container/llama"
        assert source.is_huggingface is False

    def test_r2_source_basic(self):
        """R2 source should work without revision."""
        source = WeightsSource(
            source="r2://account_id.bucket/models/llama",
            mount_location="/models/llama",
            auth_secret_name="r2_credentials",
        )
        assert source.source == "r2://account_id.bucket/models/llama"
        assert source.is_huggingface is False

    def test_https_source_basic(self):
        """HTTPS source should work for direct URL downloads."""
        source = WeightsSource(
            source="https://example.com/models/weights.bin",
            mount_location="/models/weights.bin",
        )
        assert source.source == "https://example.com/models/weights.bin"
        assert source.is_huggingface is False

    def test_https_source_with_auth(self):
        """HTTPS source with auth secret should work."""
        source = WeightsSource(
            source="https://private.example.com/models/weights.bin",
            mount_location="/models/weights.bin",
            auth_secret_name="http_auth_token",
        )
        assert source.source == "https://private.example.com/models/weights.bin"
        assert source.auth_secret_name == "http_auth_token"

    def test_https_source_invalid_format(self):
        """HTTPS source with invalid format should fail."""
        with pytest.raises(pydantic.ValidationError, match="Invalid HTTPS URL format"):
            WeightsSource(
                source="https:///path/only",  # Missing hostname
                mount_location="/models/weights.bin",
            )

    def test_mount_location_must_be_absolute(self):
        """mount_location must be an absolute path."""
        with pytest.raises(pydantic.ValidationError, match="must be an absolute path"):
            WeightsSource(
                source="hf://meta-llama/Llama-2-7b@main",
                mount_location="models/llama",  # Relative path - should fail
            )

    def test_cloud_storage_rejects_at_symbol(self):
        """Cloud storage sources should reject @ revision syntax."""
        with pytest.raises(
            pydantic.ValidationError,
            match="@ revision syntax is only valid for HuggingFace",
        ):
            WeightsSource(
                source="s3://my-bucket/models/llama@main",
                mount_location="/models/llama",
            )
        with pytest.raises(
            pydantic.ValidationError,
            match="@ revision syntax is only valid for HuggingFace",
        ):
            WeightsSource(
                source="gs://my-bucket/models/llama@main",
                mount_location="/models/llama",
            )
        with pytest.raises(
            pydantic.ValidationError,
            match="@ revision syntax is only valid for HuggingFace",
        ):
            WeightsSource(
                source="azure://myaccount/container/path@main",
                mount_location="/models/llama",
            )
        with pytest.raises(
            pydantic.ValidationError,
            match="@ revision syntax is only valid for HuggingFace",
        ):
            WeightsSource(
                source="r2://account_id.bucket/path@main",
                mount_location="/models/llama",
            )

    def test_source_cannot_be_empty(self):
        """source must have at least 1 character."""
        with pytest.raises(pydantic.ValidationError):
            WeightsSource(source="", mount_location="/models/llama")

    def test_hf_source_without_revision(self):
        """HuggingFace source should work without revision in URI."""
        source = WeightsSource(
            source="hf://meta-llama/Llama-2-7b", mount_location="/models/llama"
        )
        assert source.is_huggingface is True
        assert source.source == "hf://meta-llama/Llama-2-7b"

    def test_source_missing_scheme(self):
        """Source without URI scheme should error."""
        with pytest.raises(pydantic.ValidationError, match="missing a URI scheme"):
            WeightsSource(
                source="meta-llama/Llama-2-7b", mount_location="/models/llama"
            )

    def test_unsupported_uri_scheme(self):
        """Unsupported URI schemes should error."""
        with pytest.raises(
            pydantic.ValidationError, match="Unsupported source scheme 'ftp://'"
        ):
            WeightsSource(
                source="ftp://server/models/llama", mount_location="/models/llama"
            )

    def test_invalid_s3_uri_format(self):
        """S3 URI without bucket should error."""
        with pytest.raises(pydantic.ValidationError, match="Invalid S3 URI format"):
            WeightsSource(source="s3://", mount_location="/models/llama")

    def test_invalid_gs_uri_format(self):
        """GCS URI without bucket should error."""
        with pytest.raises(pydantic.ValidationError, match="Invalid GS URI format"):
            WeightsSource(source="gs://", mount_location="/models/llama")

    def test_invalid_azure_uri_format(self):
        """Azure URI without account should error."""
        with pytest.raises(pydantic.ValidationError, match="Invalid AZURE URI format"):
            WeightsSource(source="azure://", mount_location="/models/llama")

    def test_invalid_r2_uri_format(self):
        """R2 URI without bucket should error."""
        with pytest.raises(pydantic.ValidationError, match="Invalid R2 URI format"):
            WeightsSource(source="r2://", mount_location="/models/llama")

    def test_invalid_hf_uri_format(self):
        """HuggingFace URI without repo should error."""
        with pytest.raises(
            pydantic.ValidationError, match="Invalid HuggingFace URI format"
        ):
            WeightsSource(source="hf://", mount_location="/models/llama")


class TestWeights:
    """Tests for the Weights model (list of WeightsSource)."""

    def test_empty_weights(self):
        """Empty weights list should work."""
        weights = Weights([])
        assert weights.sources == []

    def test_single_hf_source(self):
        """Single HuggingFace source."""
        weights = Weights(
            [
                WeightsSource(
                    source="hf://meta-llama/Llama-2-7b@main",
                    mount_location="/models/llama",
                )
            ]
        )
        assert len(weights.sources) == 1
        assert weights.sources[0].is_huggingface is True

    def test_multi_source_weights(self):
        """Multiple sources from different providers."""
        weights = Weights(
            [
                WeightsSource(
                    source="hf://meta-llama/Llama-2-7b@main",
                    mount_location="/models/base",
                ),
                WeightsSource(
                    source="s3://my-bucket/adapters/lora",
                    mount_location="/models/adapter",
                    auth_secret_name="aws_credentials",
                ),
            ]
        )
        assert len(weights.sources) == 2
        assert weights.sources[0].is_huggingface is True
        assert weights.sources[1].is_huggingface is False

    def test_duplicate_mount_location_error(self):
        """Duplicate mount_location should error."""
        with pytest.raises(
            pydantic.ValidationError, match="Duplicate mount_location '/models/llama'"
        ):
            Weights(
                [
                    WeightsSource(
                        source="hf://meta-llama/Llama-2-7b@main",
                        mount_location="/models/llama",
                    ),
                    WeightsSource(
                        source="s3://my-bucket/adapters/lora",
                        mount_location="/models/llama",  # Duplicate - should fail
                        auth_secret_name="aws_credentials",
                    ),
                ]
            )


class TestTrussConfigWeights:
    """Tests for weights field in TrussConfig."""

    def test_empty_weights_config(self, default_config):
        """Empty weights should work."""
        config = TrussConfig(python_version="py39")
        assert config.weights.sources == []

    def test_weights_from_yaml(self, tmp_path):
        """Weights should be parsed from YAML."""
        yaml_content = """
        weights:
          - source: "hf://meta-llama/Llama-2-7b@main"
            mount_location: "/models/llama"
          - source: "s3://my-bucket/models/adapter"
            mount_location: "/models/adapter"
            auth_secret_name: "aws_credentials"
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = TrussConfig.from_yaml(config_path)
        assert len(config.weights.sources) == 2
        assert config.weights.sources[0].source == "hf://meta-llama/Llama-2-7b@main"
        assert config.weights.sources[1].source == "s3://my-bucket/models/adapter"

    def test_cannot_use_both_model_cache_and_weights(self, tmp_path):
        """Should error if both model_cache and weights are specified."""
        yaml_content = """
        model_cache:
          - repo_id: "test/model"
            use_volume: false
        weights:
          - source: "hf://meta-llama/Llama-2-7b@main"
            mount_location: "/models/llama"
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="only one of `model_cache` and `weights`"):
            TrussConfig.from_yaml(config_path)

    def test_weights_serialization_roundtrip(self, tmp_path):
        """Weights should serialize and deserialize correctly."""
        config = TrussConfig(
            python_version="py39",
            weights=Weights(
                [
                    WeightsSource(
                        source="hf://meta-llama/Llama-2-7b@main",
                        mount_location="/models/llama",
                        allow_patterns=["*.safetensors"],
                    )
                ]
            ),
        )

        out_path = tmp_path / "out.yaml"
        config.write_to_yaml_file(out_path, verbose=True)

        config_new = TrussConfig.from_yaml(out_path)
        assert len(config_new.weights.sources) == 1
        assert config_new.weights.sources[0].source == "hf://meta-llama/Llama-2-7b@main"
        assert config_new.weights.sources[0].mount_location == "/models/llama"
        assert config_new.weights.sources[0].allow_patterns == ["*.safetensors"]
