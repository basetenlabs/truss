import json
import shlex

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
        patch_kwargs={"max-num-seqs": 32, "enable-lora": True},
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
    assert "--max-num-seqs 32" in cmd
    assert "--enable-lora" in cmd


def test_vllm_configuration_minimal():
    config = VLLMConfiguration(model="facebook/opt-125m")
    cmd = config.build_start_command()
    assert cmd == "vllm serve facebook/opt-125m --port 8000 --host 0.0.0.0"


def test_vllm_shared_fields_with_trt_llm():
    from truss.base.llm_config import TrussLLMSharedConfig

    shared = TrussLLMSharedConfig(
        model="meta-llama/Llama-3-8B",
        dtype="bfloat16",
        quantization="fp8",
        tensor_parallel_size=2,
        max_model_len=8192,
        extra_args=["--foo"],
        patch_kwargs={"bar": 1},
    )
    vllm_cfg = VLLMConfiguration(
        model=shared.model, **shared.model_dump(exclude={"model"})
    )
    assert vllm_cfg.dtype == "bfloat16"
    assert vllm_cfg.tensor_parallel_size == 2
    assert vllm_cfg.extra_args == ["--foo"]


def test_vllm_auto_tp_from_accelerator():
    config = VLLMConfiguration(model="facebook/opt-125m")
    cmd = config.build_start_command(accelerator_count=4)
    assert "--tensor-parallel-size 4" in cmd
    cmd2 = config.build_start_command(accelerator_count=None)
    assert "--tensor-parallel-size" not in cmd2


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


def test_vllm_start_command_shlex_round_trips_structured_values():
    hf_overrides = {"text_config": {"sliding_window": 4096}}
    config = VLLMConfiguration(
        model="facebook/opt-125m",
        patch_kwargs={
            "hf_overrides": hf_overrides,
            "allowed_origins": ["a", "b"],
            "chat_template": "{{ messages }} with spaces",
        },
    )
    argv = shlex.split(config.build_start_command())
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == hf_overrides
    assert json.loads(argv[argv.index("--allowed-origins") + 1]) == ["a", "b"]
    assert argv[argv.index("--chat-template") + 1] == "{{ messages }} with spaces"


def test_vllm_patch_kwargs_false_bool_emits_no_flag():
    config = VLLMConfiguration(
        model="facebook/opt-125m",
        patch_kwargs={"enable_prefix_caching": False, "enforce_eager": True},
    )
    argv = shlex.split(config.build_start_command())
    assert "--no-enable-prefix-caching" in argv
    assert "--enable-prefix-caching" not in argv
    assert "--enforce-eager" in argv


def test_trt_llm_v2_max_model_len_respects_max_seq_len_bound():
    from truss.base.trt_llm_config import TRTLLMRuntimeConfigurationV2

    runtime = TRTLLMRuntimeConfigurationV2(max_model_len=8192)
    assert runtime.max_seq_len == 8192
    with pytest.raises(ValueError, match="max_model_len"):
        TRTLLMRuntimeConfigurationV2(max_model_len=2_000_000)


def test_trt_llm_v2_tensor_parallel_size_is_strict():
    from truss.base.trt_llm_config import TRTLLMRuntimeConfigurationV2

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        TRTLLMRuntimeConfigurationV2(tensor_parallel_size="2")
