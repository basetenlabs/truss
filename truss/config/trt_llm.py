import json
import logging
import warnings
from enum import Enum
from typing import Optional

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import BaseModel, PydanticDeprecatedSince20, validator
from rich.console import Console

# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class TrussTRTLLMModel(str, Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    WHISPER = "whisper"


class TrussTRTLLMQuantizationType(str, Enum):
    NO_QUANT = "no_quant"
    WEIGHTS_ONLY_INT8 = "weights_int8"
    WEIGHTS_KV_INT8 = "weights_kv_int8"
    WEIGHTS_ONLY_INT4 = "weights_int4"
    WEIGHTS_KV_INT4 = "weights_kv_int4"
    SMOOTH_QUANT = "smooth_quant"
    FP8 = "fp8"
    FP8_KV = "fp8_kv"


class TrussTRTLLMPluginConfiguration(BaseModel):
    paged_kv_cache: bool = True
    gemm_plugin: str = "auto"
    use_paged_context_fmha: bool = False
    use_fp8_context_fmha: bool = False


class CheckpointSource(str, Enum):
    HF: str = "HF"
    GCS: str = "GCS"
    LOCAL: str = "LOCAL"
    # REMOTE_URL is useful when the checkpoint lives on remote storage accessible via HTTP (e.g a presigned URL)
    REMOTE_URL: str = "REMOTE_URL"


class CheckpointRepository(BaseModel):
    source: CheckpointSource
    repo: str


class TrussTRTLLMBuildConfiguration(BaseModel):
    base_model: TrussTRTLLMModel
    max_input_len: int
    max_output_len: int
    max_batch_size: int
    max_beam_width: int = 1
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
    kv_cache_free_gpu_mem_fraction: float = 0.9
    num_builder_gpus: Optional[int] = None

    @validator("max_beam_width")
    def check_max_beam_width(cls, v: int):
        if isinstance(v, int):
            if v != 1:
                raise ValueError(
                    "max_beam_width greater than 1 is not currently supported"
                )
        return v


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
        self._validate_kv_cache_flags()
        if self.build.checkpoint_repository.source == CheckpointSource.HF:
            self._validate_hf_repo_id()

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

    def _validate_kv_cache_flags(self):
        if self.build is None:
            return self
        if not self.build.plugin_configuration.paged_kv_cache and (
            self.build.plugin_configuration.use_paged_context_fmha
            or self.build.plugin_configuration.use_fp8_context_fmha
        ):
            raise ValueError(
                "Using paged context fmha or fp8 context fmha requires requires paged kv cache"
            )
        if (
            self.build.plugin_configuration.use_fp8_context_fmha
            and not self.build.plugin_configuration.use_paged_context_fmha
        ):
            raise ValueError("Using fp8 context fmha requires paged context fmha")
        return self

    def _validate_hf_repo_id(self):
        try:
            validate_repo_id(self.build.checkpoint_repository.repo)
        except HFValidationError as e:
            raise ValueError(
                f"HuggingFace repository validation failed: {str(e)}"
            ) from e

    @property
    def requires_build(self):
        if self.build is not None:
            return True
        return False

    # TODO(Abu): Replace this with model_dump(json=True)
    # when pydantic v2 is used here
    def to_json_dict(self, verbose=True):
        return json.loads(self.json(exclude_unset=not verbose))
