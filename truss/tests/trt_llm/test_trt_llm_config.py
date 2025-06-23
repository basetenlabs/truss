import copy

import pydantic
import pytest

from truss.base.trt_llm_config import (
    TRTLLMConfiguration,
    TRTLLMConfigurationV1,
    TRTLLMConfigurationV2,
    TrussSpecDecMode,
    TrussSpeculatorConfiguration,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMRuntimeConfiguration,
)


def test_trt_llm_config_init_from_pydantic_models(trtllm_config):
    build_config = TrussTRTLLMBuildConfiguration(**trtllm_config["trt_llm"]["build"])
    config = TRTLLMConfiguration(
        build=build_config, runtime=TrussTRTLLMRuntimeConfiguration()
    )
    assert config.inference_stack == "v1"


def test_trt_llm_config_init_from_pydantic_models_v1(trtllm_config):
    build_config = TrussTRTLLMBuildConfiguration(**trtllm_config["trt_llm"]["build"])
    config = TRTLLMConfiguration(
        build=build_config, runtime=TrussTRTLLMRuntimeConfiguration()
    )
    config_py = config.model_dump(exclude_unset=False)
    assert config_py["inference_stack"] == "v1"
    assert config.inference_stack == "v1"

    config_roundtrip = TRTLLMConfiguration(**config_py)
    assert config_roundtrip.inference_stack == "v1"


def test_trt_llm_config_init_from_pydantic_models_v2(trtllm_config_v2):
    config = TRTLLMConfiguration(**trtllm_config_v2["trt_llm"])
    config_py = config.model_dump(exclude_unset=False)
    assert config_py["inference_stack"] == "v2"
    assert config.inference_stack == "v2"
    assert isinstance(config.root, TRTLLMConfigurationV2)

    config_roundtrip = TRTLLMConfiguration(**config_py)
    assert config_roundtrip.inference_stack == "v2"


def test_trt_llm_config_v2(trtllm_config_v2):
    config = TRTLLMConfigurationV2(**trtllm_config_v2["trt_llm"])
    config_root = TRTLLMConfiguration(**trtllm_config_v2["trt_llm"])
    assert config_root.model_dump() == config.model_dump()
    assert config.build.checkpoint_repository.repo == "meta/llama4-500B"
    assert config.inference_stack == "v2"


def validate_incorrect_trt_llm_config_runtime(trtllm_config, trtllm_config_v2):
    trtllm_config["trt_llm"]["runtime"] = trtllm_config_v2["trt_llm"]["runtime"]
    with pytest.raises(Exception):
        TRTLLMConfiguration(**trtllm_config["trt_llm"])


def validate_incorrect_trt_llm_config_v2_runtime(trtllm_config, trtllm_config_v2):
    trtllm_config_v2["trt_llm"]["runtime"] = trtllm_config["trt_llm"]["runtime"]
    with pytest.raises(Exception):
        TRTLLMConfiguration(**trtllm_config["trt_llm"])


def raise_v2_v1(trtllm_config, trtllm_config_v2):
    with pytest.raises(Exception):
        trtllm_config["trt_llm"]["inference_stack"] = "v2"
        TRTLLMConfiguration(**trtllm_config["trt_llm"])
    with pytest.raises(Exception):
        trtllm_config_v2["trt_llm"]["inference_stack"] = "v1"
        TRTLLMConfiguration(**trtllm_config_v2["trt_llm"])


def test_trt_llm_config_init_with_lora(trtllm_config):
    build = trtllm_config["trt_llm"]["build"]
    build["lora_adapters"] = {
        "adapter1": {"source": "HF", "repo": "meta/llama4-500B-lora2"},
        "adapter2": {"source": "HF", "repo": "meta/llama4-500B-lora1"},
    }

    build_config = TrussTRTLLMBuildConfiguration(**build)
    assert len(build_config.lora_adapters) == 2
    assert build_config.lora_adapters["adapter1"].source == "HF"

    TRTLLMConfigurationV1(build=build_config, runtime=TrussTRTLLMRuntimeConfiguration())

    with pytest.raises(Exception):
        build["lora_adapters"] = {
            "adapter-$ bad-invalid": {"source": "HF", "repo": "meta/llama4-500B-lora2"}
        }
        build_config2 = TrussTRTLLMBuildConfiguration(**build)
        print(build_config2.lora_adapters)


def test_trt_llm_configuration_init_and_migrate_deprecated_runtime_fields(
    deprecated_trtllm_config,
):
    trt_llm_config = TRTLLMConfigurationV1(**deprecated_trtllm_config["trt_llm"])
    assert trt_llm_config.runtime.model_dump(exclude_unset=True) == {
        "kv_cache_free_gpu_mem_fraction": 0.1,
        "enable_chunked_context": True,
        "batch_scheduler_policy": "max_utilization",
        "request_default_max_tokens": 10,
        "total_token_limit": 100,
    }


def test_trt_llm_encoder(trtllm_config_encoder):
    config = TRTLLMConfigurationV1(**trtllm_config_encoder["trt_llm"])
    # no paged_kv_cache for encoder and no use_paged_context_fmha
    assert config.build.plugin_configuration.paged_kv_cache is False
    assert config.build.plugin_configuration.use_paged_context_fmha is False


def test_trt_llm_encoder_autoconfig(trtllm_config_encoder):
    trt_llm_config = TRTLLMConfigurationV1(**trtllm_config_encoder["trt_llm"])
    try:
        from transformers import AutoConfig
    except ImportError:
        pytest.skip("transformers is not installed")

    try:
        AutoConfig.from_pretrained(trt_llm_config.build.checkpoint_repository.repo)
    except Exception:
        pytest.skip("checkpoint not found - huggingface must be down.")

    assert (
        trt_llm_config.model_dump(mode="json")["runtime"]["webserver_default_route"]
        == "/v1/embeddings"
    )


def test_trt_llm_chunked_prefill_fix(trtllm_config):
    """make sure that the chunked prefill validation is working"""
    trt_llm_config = TRTLLMConfigurationV1(**trtllm_config["trt_llm"])

    # check that the default is True
    assert trt_llm_config.build.plugin_configuration.paged_kv_cache is True
    assert trt_llm_config.build.plugin_configuration.use_paged_context_fmha is True
    assert trt_llm_config.runtime.enable_chunked_context is True

    # fixed for user
    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.build.plugin_configuration.paged_kv_cache = False
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    with pytest.raises(pydantic.ValidationError):
        trt_llm_fixed = TRTLLMConfigurationV1(**trt_llm2.model_dump())

    # fixed for user
    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    with pytest.raises(pydantic.ValidationError):
        trt_llm_fixed = TRTLLMConfigurationV1(**trt_llm2.model_dump())

    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.runtime.enable_chunked_context = False
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    trt_llm2.build.plugin_configuration.paged_kv_cache = False

    trt_llm_fixed = TRTLLMConfigurationV1(**trt_llm2.model_dump())
    assert trt_llm_fixed.build.plugin_configuration.use_paged_context_fmha is False
    assert trt_llm_fixed.runtime.enable_chunked_context is False


def test_trt_llm_lookahead_decoding(trtllm_config):
    trt_llm_config = TRTLLMConfigurationV1(**trtllm_config["trt_llm"])

    with pytest.raises(ValueError):
        trt_llm_config.build.speculator = TrussSpeculatorConfiguration(
            speculative_decoding_mode=TrussSpecDecMode.LOOKAHEAD_DECODING,
            num_draft_tokens=None,
            lookahead_windows_size=None,
            lookahead_ngram_size=None,
            lookahead_verification_set_size=None,
        )
        # need to specify lookahead_windows_size and lookahead_ngram_size and lookahead_verification_set_size
        TRTLLMConfigurationV1(**trt_llm_config.model_dump())

    trt_llm_config.build.speculator = TrussSpeculatorConfiguration(
        speculative_decoding_mode=TrussSpecDecMode.LOOKAHEAD_DECODING,
        num_draft_tokens=None,  # will be overwriten
        lookahead_windows_size=10,
        lookahead_ngram_size=10,
        lookahead_verification_set_size=10,
    )
    with_spec = TRTLLMConfigurationV1(**trt_llm_config.model_dump())
    assert with_spec.build.speculator.lookahead_windows_size == 10
    assert with_spec.build.speculator.lookahead_ngram_size == 10
    assert with_spec.build.speculator.lookahead_verification_set_size == 10
    assert (
        with_spec.build.speculator.speculative_decoding_mode
        == TrussSpecDecMode.LOOKAHEAD_DECODING
    )
    assert with_spec.build.speculator.num_draft_tokens == 179

    with pytest.raises(ValueError):
        trt_llm_config.build.speculator = TrussSpeculatorConfiguration(
            speculative_decoding_mode=TrussSpecDecMode.LOOKAHEAD_DECODING,
            num_draft_tokens=None,
            lookahead_windows_size=100,
            lookahead_ngram_size=100,
            lookahead_verification_set_size=100,
        )
        # need to specify num_draft_tokens
        TRTLLMConfigurationV1(**trt_llm_config.model_dump())
        # will lead to ValueError -> too many draft tokens are generated with 100 lookahead windows
