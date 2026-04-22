import pytest

from truss.base.vllm_config import VLLMConfiguration


def test_vllm_configuration_build_start_command():
    config = VLLMConfiguration(
        model="meta-llama/Llama-2-7b-hf",
        port=8000,
        host="0.0.0.0",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="bfloat16",
        quantization="fp8",
        trust_remote_code=True,
        served_model_name="llama-2-7b",
        extra_args=["--enable-prefix-caching"],
    )
    cmd = config.build_start_command()
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "--port 8000" in cmd
    assert "--host 0.0.0.0" in cmd
    assert "--tensor-parallel-size 1" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd
    assert "--max-model-len 4096" in cmd
    assert "--dtype bfloat16" in cmd
    assert "--quantization fp8" in cmd
    assert "--served-model-name llama-2-7b" in cmd
    assert "--trust-remote-code" in cmd
    assert "--enable-prefix-caching" in cmd


def test_vllm_configuration_minimal():
    config = VLLMConfiguration(model="facebook/opt-125m")
    cmd = config.build_start_command()
    assert cmd == "vllm serve facebook/opt-125m --port 8000 --host 0.0.0.0"


def test_vllm_gpu_memory_utilization_validation():
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        VLLMConfiguration(model="test", gpu_memory_utilization=1.5)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        VLLMConfiguration(model="test", gpu_memory_utilization=0.0)


def test_vllm_config_in_truss_config():
    from truss.base.truss_config import TrussConfig

    config = TrussConfig(vllm=VLLMConfiguration(model="meta-llama/Llama-2-7b-hf"))
    assert config.vllm is not None
    assert config.vllm.model == "meta-llama/Llama-2-7b-hf"


def test_vllm_and_trt_llm_mutual_exclusion():
    from truss.base.trt_llm_config import TRTLLMConfiguration
    from truss.base.truss_config import TrussConfig

    with pytest.raises(ValueError, match="vllm and trt_llm cannot both be configured"):
        TrussConfig(
            vllm=VLLMConfiguration(model="test"),
            trt_llm=TRTLLMConfiguration(
                build={
                    "base_model": "decoder",
                    "checkpoint_repository": {"source": "HF", "repo": "test"},
                }
            ),
        )
