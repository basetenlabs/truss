import copy
import tempfile
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pydantic
import pytest
import yaml

from truss.base.trt_llm_config import TrussTRTLLMQuantizationType
from truss.base.truss_config import (
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    Accelerator,
    AcceleratorSpec,
    BaseImage,
    Build,
    CacheInternal,
    DockerAuthSettings,
    DockerAuthType,
    ModelCache,
    ModelRepo,
    Resources,
    TrussConfig,
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
                accelerator=AcceleratorSpec(accelerator=Accelerator.V100, count=1),
                use_gpu=True,
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
            Resources(
                accelerator=AcceleratorSpec(accelerator=Accelerator.T4, count=1),
                use_gpu=True,
            ),
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
                accelerator=AcceleratorSpec(accelerator=Accelerator.A10G, count=4),
                use_gpu=True,
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
            [ModelRepo(repo_id="test/model"), ModelRepo(repo_id="test/model2")]
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
        python_version="py39", model_cache=ModelCache([ModelRepo(repo_id="test/model")])
    )

    new_config = default_config
    new_config["model_cache"] = [{"repo_id": "test/model"}]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["model_cache"][0].get("revision") is None


def test_huggingface_cache_single_model_non_default_revision_v1():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        model_cache=ModelCache([ModelRepo(repo_id="test/model", revision="not-main")]),
    )

    assert config.to_dict(verbose=False)["model_cache"][0].get("revision") == "not-main"


def test_huggingface_cache_multiple_models_default_revision(default_config):
    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(
            [
                ModelRepo(repo_id="test/model1", revision="main"),
                ModelRepo(repo_id="test/model2"),
            ]
        ),
    )

    new_config = default_config
    new_config["model_cache"] = [
        {"repo_id": "test/model1", "revision": "main"},
        {"repo_id": "test/model2"},
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["model_cache"], config.to_dict(verbose=True)[
        "model_cache"
    ]
    assert config.to_dict(verbose=True)["model_cache"][0].get("revision") == "main"
    assert config.to_dict(verbose=True)["model_cache"][1].get("revision") is None


def test_huggingface_cache_multiple_models_mixed_revision(default_config):
    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(
            [
                ModelRepo(repo_id="test/model1"),
                ModelRepo(repo_id="test/model2", revision="not-main2"),
            ]
        ),
    )

    new_config = default_config
    new_config["model_cache"] = [
        {"repo_id": "test/model1"},
        {"repo_id": "test/model2", "revision": "not-main2"},
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["model_cache"][0].get("revision") is None
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

    del trtllm_config["trt_llm"]["build"]["quantization_type"]
    with pytest.raises(ValueError):
        TrussConfig.model_validate(trtllm_config)


@pytest.mark.parametrize("verbose, expect_equal", [(False, True), (True, False)])
def test_to_dict_trtllm(verbose, expect_equal, trtllm_config):
    assert (
        TrussConfig.model_validate(trtllm_config).to_dict(verbose=verbose)
        == trtllm_config
    ) == expect_equal


@pytest.mark.parametrize("verbose, expect_equal", [(False, True), (True, False)])
def test_to_dict_trtllm_spec_dec(verbose, expect_equal, trtllm_spec_dec_config):
    assert (
        TrussConfig.model_validate(trtllm_spec_dec_config).to_dict(verbose=verbose)
        == trtllm_spec_dec_config
    ) == expect_equal


@pytest.mark.parametrize("verbose, expect_equal", [(False, True), (True, False)])
def test_to_dict_trtllm_spec_dec_full(
    verbose, expect_equal, trtllm_spec_dec_config_full
):
    assert (
        TrussConfig.model_validate(trtllm_spec_dec_config_full).to_dict(verbose=verbose)
        == trtllm_spec_dec_config_full
    ) == expect_equal


@pytest.mark.parametrize("should_raise", [False, True])
def test_model_validate_spec_dec_trt_llm(should_raise, trtllm_spec_dec_config):
    test_config = copy.deepcopy(trtllm_spec_dec_config)
    if should_raise:
        test_config["trt_llm"]["build"]["speculator"]["speculative_decoding_mode"] = (
            None
        )
        with pytest.raises(ValueError):
            TrussConfig.model_validate(test_config)
        test_config["trt_llm"]["build"]["speculator"]["checkpoint_repository"] = None
        with pytest.raises(ValueError):
            TrussConfig.model_validate(test_config)
        test_config["trt_llm"]["build"]["speculator"]["checkpoint_repository"] = (
            trtllm_spec_dec_config["trt_llm"]["build"]["speculator"][
                "checkpoint_repository"
            ]
        )
        test_config["trt_llm"]["build"]["plugin_configuration"][
            "use_paged_context_fmha"
        ] = False
        with pytest.raises(ValueError):
            TrussConfig.model_validate(test_config)
        test_config["trt_llm"]["build"]["plugin_configuration"][
            "use_paged_context_fmha"
        ] = True
        test_config["trt_llm"]["build"]["speculator"]["speculative_decoding_mode"] = (
            trtllm_spec_dec_config["trt_llm"]["build"]["speculator"][
                "speculative_decoding_mode"
            ]
        )
        test_config["trt_llm"]["build"]["speculator"]["num_draft_tokens"] = None
        with pytest.raises(ValueError):
            TrussConfig.model_validate(test_config)
    else:
        TrussConfig.model_validate(trtllm_spec_dec_config)


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


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [
        ("py38", "py38"),
        ("py39", "py39"),
        ("py310", "py310"),
        ("py311", "py311"),
        ("py312", "py311"),
    ],
)
def test_map_to_supported_python_version(python_version, expected_python_version):
    out_python_version = _map_to_supported_python_version(python_version)
    assert out_python_version == expected_python_version


def test_not_supported_python_minor_versions():
    with pytest.raises(
        ValueError,
        match="Mapping python version 3.6 to 3.8, "
        "the lowest version that Truss currently supports.",
    ):
        _map_to_supported_python_version("py36")
    with pytest.raises(
        ValueError,
        match="Mapping python version 3.7 to 3.8, "
        "the lowest version that Truss currently supports.",
    ):
        _map_to_supported_python_version("py37")


def test_not_supported_python_major_versions():
    with pytest.raises(NotImplementedError, match="Only python version 3 is supported"):
        _map_to_supported_python_version("py211")
