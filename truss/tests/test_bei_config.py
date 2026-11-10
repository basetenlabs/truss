import pydantic
import pytest

from truss.base.bei_config import (
    BEIBackend,
    BEIConfiguration,
    BEIImageVersions,
    BEIModelConfiguration,
    BEIQuantization,
)


def _model(**overrides) -> dict:
    data = {
        "checkpoint": "baai/bge-small-en-v1.5",
        "revision": "main",
        "served_model_name": "baai-small",
        "quantization": "none",
    }
    data.update(overrides)
    return data


def _config(**overrides) -> dict:
    data = {"backend": "nvidia", "configuration": [_model()]}
    data.update(overrides)
    return data


def test_configuration_roundtrip():
    config = BEIConfiguration.model_validate(
        _config(
            version_override="1.2.3",
            configuration=[
                _model(),
                _model(
                    checkpoint="BAAI/bge-base-en-v1.5",
                    revision=None,
                    served_model_name=None,
                    quantization="fp8",
                ),
            ],
        )
    )

    assert config.backend == BEIBackend.NVIDIA
    assert config.configuration[0].quantization == BEIQuantization.NONE
    assert config.configuration[1].quantization == BEIQuantization.FP8
    assert config.configuration[1].revision is None
    assert config.configuration[1].served_model_name is None
    assert (
        config.configuration[1].effective_served_model_name == "BAAI/bge-base-en-v1.5"
    )
    assert config.image_override == "baseten/bei:1.2.3-nvidia"

    dumped = config.model_dump(mode="json")
    assert dumped["backend"] == "nvidia"
    assert dumped["version_override"] == "1.2.3"
    assert dumped["configuration"][1]["quantization"] == "fp8"

    roundtripped = BEIConfiguration.model_validate(dumped)
    assert roundtripped == config
    assert roundtripped.to_dict() == config.to_dict()


@pytest.mark.parametrize(
    "configuration",
    [
        [
            {"checkpoint": "baai/first", "served_model_name": "shared"},
            {"checkpoint": "baai/second", "served_model_name": "shared"},
        ],
        [{"checkpoint": "baai/same"}, {"checkpoint": "baai/same"}],
    ],
)
def test_duplicate_effective_served_names(configuration):
    with pytest.raises(
        pydantic.ValidationError, match="Duplicate effective served_model_name"
    ):
        BEIConfiguration(backend=BEIBackend.NVIDIA, configuration=configuration)


def test_duplicate_effective_name_from_default_checkpoint():
    with pytest.raises(
        pydantic.ValidationError, match="Duplicate effective served_model_name"
    ):
        BEIConfiguration(
            backend=BEIBackend.NVIDIA,
            configuration=[
                {"checkpoint": "baai/first"},
                {"checkpoint": "baai/second", "served_model_name": "baai/first"},
            ],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint", "baai/model;other"),
        ("checkpoint", "baai/model\nother"),
        ("checkpoint", "baai/model\x00other"),
        ("checkpoint", ""),
        ("checkpoint", "   "),
        ("checkpoint", "baai/model with space"),
        ("revision", "main;other"),
        ("revision", "main\nother"),
        ("revision", "main\x00other"),
        ("revision", ""),
        ("revision", "none"),
        ("served_model_name", "name;other"),
        ("served_model_name", "name\nother"),
        ("served_model_name", "name\x00other"),
        ("served_model_name", ""),
        ("served_model_name", "none"),
    ],
)
def test_invalid_model_values(field, value):
    with pytest.raises(pydantic.ValidationError):
        BEIModelConfiguration(**_model(**{field: value}))


def test_local_checkpoint_still_rejects_multi_value_separator():
    with pytest.raises(pydantic.ValidationError, match="multi-value delimiter"):
        BEIModelConfiguration(checkpoint="/opt/models/first;/opt/models/second")


@pytest.mark.parametrize("quantization", ["mxfp8", "nvfp4", "MXFP8", "NVFP4"])
def test_unavailable_quantization_modes(quantization):
    with pytest.raises(pydantic.ValidationError, match="not available in BEIv3"):
        BEIModelConfiguration(**_model(quantization=quantization))


def test_tpu_rejects_fp8_quantization():
    with pytest.raises(
        pydantic.ValidationError, match="not supported with the TPU backend"
    ):
        BEIConfiguration(
            backend=BEIBackend.TPU, configuration=[_model(quantization="fp8")]
        )


def test_tpu_rejects_multiple_models():
    with pytest.raises(
        pydantic.ValidationError, match="Multiple models are not supported with the TPU"
    ):
        BEIConfiguration(
            backend=BEIBackend.TPU,
            configuration=[
                _model(),
                _model(checkpoint="baai/second", served_model_name="second"),
            ],
        )


def test_tpu_accepts_single_unquantized_model():
    config = BEIConfiguration(
        backend=BEIBackend.TPU, configuration=[_model(quantization="none")]
    )

    assert config.backend == BEIBackend.TPU
    assert config.image_override is None


@pytest.mark.parametrize(
    "data,match",
    [
        ({"backend": "nvidia"}, "configuration"),
        ({"backend": "nvidia", "configuration": []}, "at least 1 item"),
        ({"configuration": [_model()]}, "backend"),
    ],
)
def test_missing_required_configuration(data, match):
    with pytest.raises(pydantic.ValidationError, match=match):
        BEIConfiguration.model_validate(data)


@pytest.mark.parametrize(
    "version_override",
    ["", "v1:latest", "registry.example.com/bei:v1", "-v1", ".v1", "v1 branch"],
)
def test_invalid_version_override(version_override):
    with pytest.raises(pydantic.ValidationError):
        BEIConfiguration(
            backend="nvidia",
            version_override=version_override,
            configuration=[_model()],
        )


def test_version_override_length_includes_backend_suffix():
    max_nvidia_prefix = "v" + "a" * 120
    config = BEIConfiguration(
        backend="nvidia", version_override=max_nvidia_prefix, configuration=[_model()]
    )
    assert config.image_override is not None
    assert len(config.image_override.split(":", 1)[1]) == 128

    with pytest.raises(pydantic.ValidationError, match="maximum is 128"):
        BEIConfiguration(
            backend="nvidia",
            version_override=max_nvidia_prefix + "a",
            configuration=[_model()],
        )

    max_tpu_prefix = "v" + "a" * 123
    tpu_config = BEIConfiguration(
        backend="tpu", version_override=max_tpu_prefix, configuration=[_model()]
    )
    assert tpu_config.image_override is not None
    assert len(tpu_config.image_override.split(":", 1)[1]) == 128

    with pytest.raises(pydantic.ValidationError, match="maximum is 128"):
        BEIConfiguration(
            backend="tpu",
            version_override=max_tpu_prefix + "a",
            configuration=[_model()],
        )


def test_nested_forbid_extra():
    config = BEIConfiguration.model_validate({**_config(), "unexpected": True})
    with pytest.raises(pydantic.ValidationError, match="Extra fields not allowed"):
        config.validate_forbid_extra()

    nested_model_extra = BEIConfiguration.model_validate(
        _config(configuration=[{**_model(), "unexpected": True}])
    )
    with pytest.raises(pydantic.ValidationError, match="Extra fields not allowed"):
        nested_model_extra.validate_forbid_extra()

    nested_images_extra = BEIImageVersions.model_validate(
        {"nvidia_image": "baseten/bei:1.0-nvidia", "unexpected": True}
    )
    with pytest.raises(pydantic.ValidationError, match="Extra fields not allowed"):
        nested_images_extra.validate_forbid_extra()
