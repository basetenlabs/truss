import pytest
import yaml
from truss.truss_config import (
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    DEFAULT_USE_GPU,
    Accelerator,
    AcceleratorSpec,
    BaseImage,
    HuggingFaceCache,
    HuggingFaceModel,
    Resources,
    Train,
    TrussConfig,
)
from truss.types import ModelFrameworkType


@pytest.mark.parametrize(
    "input_dict, expect_resources, output_dict",
    [
        (
            {},
            Resources(),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": DEFAULT_USE_GPU,
                "accelerator": None,
            },
        ),
        (
            {"accelerator": None},
            Resources(),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": DEFAULT_USE_GPU,
                "accelerator": None,
            },
        ),
        (
            {"accelerator": "V100"},
            Resources(accelerator=AcceleratorSpec(Accelerator.V100, 1), use_gpu=True),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": True,
                "accelerator": "V100",
            },
        ),
        (
            {"accelerator": "T4:1"},
            Resources(accelerator=AcceleratorSpec(Accelerator.T4, 1), use_gpu=True),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": True,
                "accelerator": "T4",
            },
        ),
        (
            {"accelerator": "A10G:4"},
            Resources(accelerator=AcceleratorSpec(Accelerator.A10G, 4), use_gpu=True),
            {
                "cpu": DEFAULT_CPU,
                "memory": DEFAULT_MEMORY,
                "use_gpu": True,
                "accelerator": "A10G:4",
            },
        ),
    ],
)
def test_parse_resources(input_dict, expect_resources, output_dict):
    parsed_result = Resources.from_dict(input_dict)
    assert parsed_result == expect_resources
    assert parsed_result.to_dict() == output_dict


@pytest.mark.parametrize(
    "input_str, expected_acc",
    [
        (None, AcceleratorSpec(None, 0)),
        ("T4", AcceleratorSpec(Accelerator.T4, 1)),
        ("A10G:4", AcceleratorSpec(Accelerator.A10G, 4)),
        ("A100:8", AcceleratorSpec(Accelerator.A100, 8)),
    ],
)
def test_acc_spec_from_str(input_str, expected_acc):
    assert AcceleratorSpec.from_str(input_str) == expected_acc


@pytest.mark.parametrize(
    "input_dict, expect_base_image, output_dict",
    [
        (
            {},
            BaseImage(),
            {
                "image": "",
                "python_executable_path": "",
            },
        ),
        (
            {"image": "custom_base_image", "python_executable_path": "/path/python"},
            BaseImage(image="custom_base_image", python_executable_path="/path/python"),
            {
                "image": "custom_base_image",
                "python_executable_path": "/path/python",
            },
        ),
    ],
)
def test_parse_base_image(input_dict, expect_base_image, output_dict):
    parsed_result = BaseImage.from_dict(input_dict)
    assert parsed_result == expect_base_image
    assert parsed_result.to_dict() == output_dict


def generate_default_config():
    config = {
        "environment_variables": {},
        "external_package_dirs": [],
        "model_metadata": {},
        "model_name": None,
        "python_version": "py39",
        "requirements": [],
        "resources": {
            "accelerator": None,
            "cpu": "500m",
            "memory": "512Mi",
            "use_gpu": False,
        },
        "secrets": {},
        "system_packages": [],
    }
    return config


def test_default_config_not_crowded_end_to_end():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
    )

    config_yaml = """environment_variables: {}
external_package_dirs: []
model_metadata: {}
model_name: null
python_version: py39
requirements: []
resources:
  accelerator: null
  cpu: 500m
  memory: 512Mi
  use_gpu: false
secrets: {}
system_packages: []
"""

    assert config_yaml.strip() == yaml.dump(config.to_dict(verbose=False)).strip()


@pytest.mark.parametrize(
    "model_framework",
    [ModelFrameworkType.CUSTOM, ModelFrameworkType.SKLEARN, ModelFrameworkType.PYTORCH],
)
def test_model_framework(model_framework):
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        model_framework=model_framework,
    )

    new_config = generate_default_config()
    if model_framework == ModelFrameworkType.CUSTOM:
        assert new_config == config.to_dict(verbose=False)
    else:
        new_config["model_framework"] = model_framework.value
        assert new_config == config.to_dict(verbose=False)


def test_non_default_train():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        train=Train(resources=Resources(use_gpu=True, accelerator="A10G")),
    )

    updated_train = {
        "resources": {
            "accelerator": "A10G",
            "cpu": "500m",
            "memory": "512Mi",
            "use_gpu": True,
        },
        "variables": {},
    }

    new_config = generate_default_config()
    new_config["train"] = updated_train

    assert new_config == config.to_dict(verbose=False)


def test_huggingface_cache_single_model_default_revision():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        hf_cache=HuggingFaceCache(models=[HuggingFaceModel("test/model")]),
    )

    new_config = generate_default_config()
    new_config["hf_cache"] = [
        {
            "repo_id": "test/model",
        }
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["hf_cache"][0].get("revision") is None


def test_huggingface_cache_single_model_non_default_revision():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        hf_cache=HuggingFaceCache(models=[HuggingFaceModel("test/model", "not-main")]),
    )

    assert config.to_dict(verbose=False)["hf_cache"][0].get("revision") == "not-main"


def test_huggingface_cache_multiple_models_default_revision():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        hf_cache=HuggingFaceCache(
            models=[
                HuggingFaceModel("test/model1", "main"),
                HuggingFaceModel("test/model2"),
            ]
        ),
    )

    new_config = generate_default_config()
    new_config["hf_cache"] = [
        {"repo_id": "test/model1", "revision": "main"},
        {
            "repo_id": "test/model2",
        },
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["hf_cache"][0].get("revision") == "main"
    assert config.to_dict(verbose=True)["hf_cache"][1].get("revision") is None


def test_huggingface_cache_multiple_models_mixed_revision():
    config = TrussConfig(
        python_version="py39",
        requirements=[],
        hf_cache=HuggingFaceCache(
            models=[
                HuggingFaceModel("test/model1"),
                HuggingFaceModel("test/model2", "not-main2"),
            ]
        ),
    )

    new_config = generate_default_config()
    new_config["hf_cache"] = [
        {
            "repo_id": "test/model1",
        },
        {"repo_id": "test/model2", "revision": "not-main2"},
    ]

    assert new_config == config.to_dict(verbose=False)
    assert config.to_dict(verbose=True)["hf_cache"][0].get("revision") is None
    assert config.to_dict(verbose=True)["hf_cache"][1].get("revision") == "not-main2"
