from truss.base.trt_llm_config import (
    TRTLLMConfiguration,
    TrussTRTLLMBatchSchedulerPolicy,
)


def test_trt_llm_configuration_init_and_migrate_deprecated_runtime_fields(
    deprecated_trtllm_config,
):
    trt_llm_config = TRTLLMConfiguration(**deprecated_trtllm_config["trt_llm"])
    assert trt_llm_config.runtime.model_dump() == {
        "kv_cache_free_gpu_mem_fraction": 0.1,
        "enable_chunked_context": True,
        "batch_scheduler_policy": TrussTRTLLMBatchSchedulerPolicy.MAX_UTILIZATION.value,
        "request_default_max_tokens": 10,
        "total_token_limit": 100,
    }
