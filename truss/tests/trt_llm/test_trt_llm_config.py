import copy

import pytest

from truss.base.trt_llm_config import (
    TRTLLMConfiguration,
    TrussSpecDecMode,
    TrussSpeculatorConfiguration,
    TrussTRTLLMBatchSchedulerPolicy,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMRuntimeConfiguration,
)


def test_trt_llm_config_init_from_pydantic_models(trtllm_config):
    build_config = TrussTRTLLMBuildConfiguration(**trtllm_config["trt_llm"]["build"])
    TRTLLMConfiguration(build=build_config, runtime=TrussTRTLLMRuntimeConfiguration())


def test_trt_llm_config_init_with_lora(trtllm_config):
    build = trtllm_config["trt_llm"]["build"]
    build["lora_adapters"] = {
        "adapter1": {"source": "HF", "repo": "meta/llama4-500B-lora2"},
        "adapter2": {"source": "HF", "repo": "meta/llama4-500B-lora1"},
    }

    build_config = TrussTRTLLMBuildConfiguration(**build)
    assert len(build_config.lora_adapters) == 2
    assert build_config.lora_adapters["adapter1"].source == "HF"

    TRTLLMConfiguration(build=build_config, runtime=TrussTRTLLMRuntimeConfiguration())

    with pytest.raises(Exception):
        build["lora_adapters"] = {
            "adapter-$ bad-invalid": {"source": "HF", "repo": "meta/llama4-500B-lora2"}
        }
        build_config2 = TrussTRTLLMBuildConfiguration(**build)
        print(build_config2.lora_adapters)


def test_trt_llm_configuration_init_and_migrate_deprecated_runtime_fields(
    deprecated_trtllm_config,
):
    trt_llm_config = TRTLLMConfiguration(**deprecated_trtllm_config["trt_llm"])
    assert trt_llm_config.runtime.model_dump() == {
        "kv_cache_free_gpu_mem_fraction": 0.1,
        "kv_cache_host_memory_bytes": None,
        "enable_chunked_context": True,
        "batch_scheduler_policy": TrussTRTLLMBatchSchedulerPolicy.MAX_UTILIZATION.value,
        "request_default_max_tokens": 10,
        "total_token_limit": 50,
        "served_model_name": None,
        "webserver_default_route": None,
    }


def test_trt_llm_encoder(trtllm_config_encoder):
    config = TRTLLMConfiguration(**trtllm_config_encoder["trt_llm"])
    # no paged_kv_cache for encoder and no use_paged_context_fmha
    assert config.build.plugin_configuration.paged_kv_cache is False
    assert config.build.plugin_configuration.use_paged_context_fmha is False


def test_trt_llm_encoder_autoconfig(trtllm_config_encoder):
    trt_llm_config = TRTLLMConfiguration(**trtllm_config_encoder["trt_llm"])
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
    trt_llm_config = TRTLLMConfiguration(**trtllm_config["trt_llm"])

    # check that the default is True
    assert trt_llm_config.build.plugin_configuration.paged_kv_cache is True
    assert trt_llm_config.build.plugin_configuration.use_paged_context_fmha is True
    assert trt_llm_config.runtime.enable_chunked_context is True

    # fixed for user
    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.build.plugin_configuration.paged_kv_cache = False
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    trt_llm_fixed = TRTLLMConfiguration(**trt_llm2.model_dump())
    print(trt_llm_fixed.build)
    assert trt_llm_fixed.build.plugin_configuration.paged_kv_cache is True

    # fixed for user
    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    trt_llm_fixed = TRTLLMConfiguration(**trt_llm2.model_dump())
    assert trt_llm_fixed.build.plugin_configuration.use_paged_context_fmha is True

    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.runtime.enable_chunked_context = False
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    trt_llm2.build.plugin_configuration.paged_kv_cache = False
    trt_llm_fixed = TRTLLMConfiguration(**trt_llm2.model_dump())
    assert trt_llm_fixed.build.plugin_configuration.use_paged_context_fmha is False
    assert trt_llm_fixed.runtime.enable_chunked_context is False


def test_trt_llm_lookahead_decoding(trtllm_config):
    trt_llm_config = TRTLLMConfiguration(**trtllm_config["trt_llm"])

    with pytest.raises(ValueError):
        trt_llm_config.build.speculator = TrussSpeculatorConfiguration(
            speculative_decoding_mode=TrussSpecDecMode.LOOKAHEAD_DECODING,
            num_draft_tokens=None,
            lookahead_windows_size=None,
            lookahead_ngram_size=None,
            lookahead_verification_set_size=None,
        )
        # need to specify lookahead_windows_size and lookahead_ngram_size and lookahead_verification_set_size
        TRTLLMConfiguration(**trt_llm_config.model_dump())

    trt_llm_config.build.speculator = TrussSpeculatorConfiguration(
        speculative_decoding_mode=TrussSpecDecMode.LOOKAHEAD_DECODING,
        num_draft_tokens=None,  # will be overwriten
        lookahead_windows_size=10,
        lookahead_ngram_size=10,
        lookahead_verification_set_size=10,
    )
    with_spec = TRTLLMConfiguration(**trt_llm_config.model_dump())
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
        TRTLLMConfiguration(**trt_llm_config.model_dump())
        # will lead to ValueError -> too many draft tokens are generated with 100 lookahead windows
