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
    }


def test_trt_llm_configuration_init_and_migrate_deprecated_runtime_fields_existing_runtime(
    deprecated_trtllm_config_with_runtime_existing,
):
    trt_llm_config = TRTLLMConfiguration(
        **deprecated_trtllm_config_with_runtime_existing["trt_llm"]
    )
    assert trt_llm_config.runtime.model_dump() == {
        "kv_cache_free_gpu_mem_fraction": 0.1,
        "kv_cache_host_memory_bytes": None,
        "enable_chunked_context": True,
        "batch_scheduler_policy": TrussTRTLLMBatchSchedulerPolicy.MAX_UTILIZATION.value,
        "request_default_max_tokens": 10,
        "total_token_limit": 100,
    }


def test_trt_llm_chunked_prefill_fix(trtllm_config):
    """make sure that the chunked prefill validation is working"""
    trt_llm_config = TRTLLMConfiguration(**trtllm_config["trt_llm"])

    assert trt_llm_config.build.plugin_configuration.paged_kv_cache is True
    assert trt_llm_config.build.plugin_configuration.use_paged_context_fmha is True
    assert trt_llm_config.runtime.enable_chunked_context is True

    with pytest.raises(ValueError):
        trt_llm2 = copy.deepcopy(trt_llm_config)
        trt_llm2.build.plugin_configuration.paged_kv_cache = False
        TRTLLMConfiguration(**trt_llm2.model_dump())

    with pytest.raises(
        ValueError
    ):  # verify you cant disable paged context fmha without disabling enable_chunked_context
        trt_llm2 = copy.deepcopy(trt_llm_config)
        trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
        TRTLLMConfiguration(**trt_llm2.model_dump())

    trt_llm2 = copy.deepcopy(trt_llm_config)
    trt_llm2.runtime.enable_chunked_context = False
    trt_llm2.build.plugin_configuration.use_paged_context_fmha = False
    TRTLLMConfiguration(**trt_llm2.model_dump())


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
