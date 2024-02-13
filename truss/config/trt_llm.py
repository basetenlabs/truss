import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TRTLLMModelArchitecture(Enum):
    LLAMA: str = "llama"
    MISTRAL: str = "mistral"


class TRTLLMQuantizationType(Enum):
    NO_QUANT: str = "no_quant"
    WEIGHTS_ONLY: str = "weights_only"
    WEIGHTS_KV_INT8: str = "weights_kv_int8"
    SMOOTH_QUANT: str = "smooth_quant"


class TRTLLMBuildConfiguration(BaseModel):
    huggingface_ckpt_repository: str
    base_model_architecture: TRTLLMModelArchitecture
    max_input_len: int
    max_output_len: int
    max_batch_size: int
    quantization_type: TRTLLMQuantizationType = TRTLLMQuantizationType.NO_QUANT
    gather_all_token_logits: bool = False
    multi_block_mode: bool = False
    calibrate_kv_cache: bool = False


class TRTLLMConfiguration(BaseModel):
    engine_repository: Optional[str]
    tokenizer_repository: Optional[str]
    tensor_parallel_count: Optional[int] = 1
    pipeline_parallel_count: Optional[int] = 1
    build_configuration: Optional[TRTLLMBuildConfiguration] = None

    @model_validator(mode="after")
    def check_minimum_required_configuration(self):
        serve_engine_configuration = (
            self.engine_repository is not None and self.tokenizer_repository is not None
        )
        build_engine_configuration = self.build_configuration is not None
        if serve_engine_configuration and build_engine_configuration:
            logger.warning(
                "Both serve and build configurations are provided. Serve configuration will be used."
            )
        if (self.engine_repository is None) ^ (self.tokenizer_repository is None):
            raise ValueError(
                "Both engine_repository and tokenizer_repository must be provided"
            )

    @property
    def requires_build(self):
        if self.engine_repository is None:
            return True
        return False
