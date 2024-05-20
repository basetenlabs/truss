import json
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrussTRTLLMModel(Enum):
    LLAMA: str = "llama"
    MISTRAL: str = "mistral"
    DEEPSEEK: str = "deepseek"


class TrussTRTLLMQuantizationType(str, Enum):
    NO_QUANT: str = "no_quant"
    WEIGHTS_ONLY_INT8: str = "weights_int8"
    WEIGHTS_KV_INT8: str = "weights_kv_int8"
    WEIGHTS_ONLY_INT4: str = "weights_int4"
    WEIGHTS_KV_INT4: str = "weights_kv_int4"
    SMOOTH_QUANT: str = "smooth_quant"
    FP8: str = "fp8"
    FP8_KV: str = "fp8_kv"


class TrussTRTLLMPluginConfiguration(BaseModel):
    multi_block_mode: bool = False
    paged_kv_cache: bool = True
    gemm_plugin: str = "float16"


class CheckpointSource(Enum):
    HF: str = "HF"
    GCS: str = "GCS"
    LOCAL: str = "LOCAL"


class CheckpointRepository(BaseModel):
    source: CheckpointSource
    repo: str


class TrussTRTLLMBuildConfiguration(BaseModel):
    base_model: TrussTRTLLMModel
    max_input_len: int
    max_output_len: int
    max_batch_size: int
    max_beam_width: int
    max_prompt_embedding_table_size: int = 0
    checkpoint_repository: CheckpointRepository
    gather_all_token_logits: bool = False
    strongly_typed: bool = False
    quantization_type: TrussTRTLLMQuantizationType = (
        TrussTRTLLMQuantizationType.NO_QUANT
    )
    tensor_parallel_count: int = 1
    pipeline_parallel_count: int = 1
    plugin_configuration: TrussTRTLLMPluginConfiguration = (
        TrussTRTLLMPluginConfiguration()
    )
    use_fused_mlp: bool = False

    class Config:
        json_encoders = {
            TrussTRTLLMModel: lambda x: x.value,
            TrussTRTLLMQuantizationType: lambda x: x.value,
            CheckpointSource: lambda x: x.value,
        }


class TrussTRTLLMServingConfiguration(BaseModel):
    engine_repository: str
    tokenizer_repository: str
    tensor_parallel_count: int = 1
    pipeline_parallel_count: int = 1


class TRTLLMConfiguration(BaseModel):
    serve: Optional[TrussTRTLLMServingConfiguration] = None
    build: Optional[TrussTRTLLMBuildConfiguration] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_minimum_required_configuration()

    # In pydantic v2 this would be `@model_validator(mode="after")` and
    # the __init__ override can be removed.
    def _validate_minimum_required_configuration(self):
        if not self.serve and not self.build:
            raise ValueError("Either serve or build configurations must be provided")
        if self.serve and self.build:
            raise ValueError("Both serve and build configurations cannot be provided")
        if self.serve is not None:
            if (self.serve.engine_repository is None) ^ (
                self.serve.tokenizer_repository is None
            ):
                raise ValueError(
                    "Both engine_repository and tokenizer_repository must be provided"
                )
        return self

    @property
    def requires_build(self):
        if self.build is not None:
            return True
        return False

    # TODO(Abu): Replace this with model_dump(json=True)
    # when pydantic v2 is used here
    def to_json_dict(self):
        return json.loads(self.json())
