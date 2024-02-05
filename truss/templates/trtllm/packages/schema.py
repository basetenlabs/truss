import numpy as np
import tritonclient
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr
from enum import Enum
from pathlib import Path

import numpy as np

class ModelInput:
    def __init__(
        self, 
        prompt: str, 
        request_id: int, 
        max_tokens: int = 50, 
        beam_width: int = 1, 
        bad_words_list: list = None, 
        stop_words_list: list = None, 
        repetition_penalty: float = 1.0, 
        ignore_eos: bool = False, 
        stream: bool = True, 
        eos_token_id: int = None
    ):
        self.prompt = prompt
        self.request_id = request_id
        self.max_tokens = max_tokens
        self.beam_width = beam_width
        self.bad_words_list = bad_words_list if bad_words_list is not None else [""]
        self.stop_words_list = stop_words_list if stop_words_list is not None else [""]
        self.repetition_penalty = repetition_penalty
        self.ignore_eos = ignore_eos
        self.stream = stream
        self.eos_token_id = eos_token_id

    def to_tensors(self):
        def prepare_grpc_tensor(name: str, input_data: np.ndarray) -> tritonclient.grpc.InferInput:
            tensor = tritonclient.grpc.InferInput(
                name, input_data.shape, tritonclient.utils.np_to_triton_dtype(input_data.dtype)
            )
            tensor.set_data_from_numpy(input_data)
            return tensor

        assert self.eos_token_id is not None or self.ignore_eos, "eos_token_id must be provided if ignore_eos is False"
        
        prompt_data = np.array([[self.prompt]], dtype=object)
        output_len_data = np.ones_like(prompt_data, dtype=np.uint32) * self.max_tokens
        bad_words_data = np.array([self.bad_words_list], dtype=object)
        stop_words_data = np.array([self.stop_words_list], dtype=object)
        stream_data = np.array([[self.stream]], dtype=bool)
        beam_width_data = np.array([[self.beam_width]], dtype=np.uint32)
        repetition_penalty_data = np.array([[self.repetition_penalty]], dtype=np.float32)

        inputs = [
            prepare_grpc_tensor("text_input", prompt_data),
            prepare_grpc_tensor("max_tokens", output_len_data),
            prepare_grpc_tensor("bad_words", bad_words_data),
            prepare_grpc_tensor("stop_words", stop_words_data),
            prepare_grpc_tensor("stream", stream_data),
            prepare_grpc_tensor("beam_width", beam_width_data),
            prepare_grpc_tensor("repetition_penalty", repetition_penalty_data),
        ]
        
        if not self.ignore_eos:
            end_id_data = np.array([[self.eos_token_id]], dtype=np.uint32)
            inputs.append(prepare_grpc_tensor("end_id", end_id_data))

        return inputs

class Quant(Enum):
    NO_QUANT = "no_quant"
    WEIGHTS_ONLY = "weights_only"
    WEIGHTS_KV_INT8 = "weights_kv_int8"
    SMOOTH_QUANT = "smooth_quant"


class EngineType(Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"


class ArgsConfig(BaseModel):
    max_input_len: Optional[int] = None
    max_output_len: Optional[int] = None
    max_batch_size: Optional[int] = None
    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    world_size: Optional[int] = None
    gather_all_token_logits: Optional[bool] = None
    multi_block_mode: Optional[bool] = None
    remove_input_padding: Optional[bool] = None
    use_gpt_attention_plugin: Optional[str] = None
    paged_kv_cache: Optional[bool] = None
    use_inflight_batching: Optional[bool] = None
    enable_context_fmha: Optional[bool] = None
    use_gemm_plugin: Optional[str] = None
    use_weight_only: Optional[bool] = None
    output_dir: Optional[str] = None
    model_dir: Optional[str] = None
    ft_model_dir: Optional[str] = None
    dtype: Optional[str] = None
    int8_kv_cache: Optional[bool] = None
    use_smooth_quant: Optional[bool] = None
    per_token: Optional[bool] = None
    per_channel: Optional[bool] = None
    parallel_build: Optional[bool] = None

    # to disable warning because `model_dir` starts with `model_` prefix
    model_config = ConfigDict(protected_namespaces=())

    def as_command_arguments(self):
        non_bool_args = [
            element
            for arg, value in self.dict().items()
            for element in [f"--{arg}", str(value)]
            if value is not None and not isinstance(value, bool)
        ]
        bool_args = [
            f"--{arg}"
            for arg, value in self.dict().items()
            if isinstance(value, bool) and value
        ]
        return non_bool_args + bool_args


class CalibrationConfig(BaseModel):
    kv_cache: Optional[bool] = None  # either to calibrate kv cache
    sq_alpha: Optional[float] = None

    def cache_path(self) -> Path:
        if self.kv_cache is not None:
            return Path("kv_cache")
        else:
            return Path(f"sq_{self.sq_alpha}")


class EngineBuildArgs(BaseModel, use_enum_values=True):
    repo: Optional[str] = None
    args: Optional[ArgsConfig] = None
    quant: Optional[Quant] = None
    calibration: Optional[CalibrationConfig] = None
    engine_type: Optional[EngineType] = None

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            repo=config["repo"],
            args=ArgsConfig(**config["args"]),
            quant=Quant(config["quant"]) if "quant" in config else None,
            calibration=CalibrationConfig(config["calibration"]) if "calibration" in config else None,
            engine_type=EngineType(config["engine_type"]) if "engine_type" in config else None,
        )

class TrussBuildConfig(BaseModel):
    """
    This is a spec for what the config.yaml looks like to take advantage of TRT-LLM + TRT-LLM builds. We structure the
    configuration with the below top-level keys.
    
    Example (for building an engine)
    ```
    build:
        model_server: TRT_LLM
        arguments:
            tokenizer_repository: "mistralai/mistral-v2-instruct"
            arguments:
                max_input_len: 1024
                max_output_len: 1024
                max_batch_size: 64
            quant: "weights_kv_int8"
            tensor_parallel_count: 2
            pipeline_parallel_count: 1
    ```
    
    Example (for using an existing engine)
    ```
    build:
        model_server: TRT_LLM
        arguments:
            engine_repository: "baseten/mistral-v2-32k"
            tensor_parallel_count: 2
            pipeline_parallel_count: 1
    ```

    """
    tokenizer_repository: str
    engine_repository: Optional[str] = None
    quant: Optional[Quant] = "no_quant"
    calibration: Optional[CalibrationConfig] = None
    engine_type: Optional[EngineType] = None
    pipeline_parallel_count: Optional[int] = 1
    tensor_parallel_count: Optional[int] = 1
    arguments: Optional[dict] = None
    _engine_build_args: Optional[EngineBuildArgs] = PrivateAttr(default=None)

    @property
    def engine_build_args(self) -> EngineBuildArgs:
        if self._engine_build_args is None:
            constructed_engine_builds_args = {
                "repo": self.tokenizer_repository,
                "quant": self.quant,
                "calibration": self.calibration,
                "engine_type": self.engine_type,
                "args": self.arguments,
            }
            
            constructed_engine_builds_args["args"]["tp_size"] = self.tensor_parallel_count
            constructed_engine_builds_args["args"]["pp_size"] = self.pipeline_parallel_count
            self._engine_build_args = EngineBuildArgs(**constructed_engine_builds_args)
        return self._engine_build_args

    @property
    def requires_build(self):
        return self.engine_repository is None