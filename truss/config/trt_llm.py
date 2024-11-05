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
    QWEN = "qwen"


class TrussTRTLLMQuantizationType(str, Enum):
    NO_QUANT = "no_quant"
    WEIGHTS_ONLY_INT8 = "weights_int8"
    WEIGHTS_KV_INT8 = "weights_kv_int8"
    WEIGHTS_ONLY_INT4 = "weights_int4"
    WEIGHTS_INT4_KV_INT8 = "weights_int4_kv_int8"
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


class TrussTRTLLMBatchSchedulerPolicy(Enum):
    MAX_UTILIZATION = 0
    GUARANTEED_NO_EVICT = 1


class TrussSpecDecMode(str, Enum):
    DRAFT_EXTERNAL: str = "DRAFT_TOKENS_EXTERNAL"


class TrussTRTLLMBuildConfiguration(BaseModel):
    base_model: TrussTRTLLMModel
    max_seq_len: int
    max_batch_size: Optional[int] = 256
    max_num_tokens: Optional[int] = 8192
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
    num_builder_gpus: Optional[int] = None
    speculative_decoding_mode: Optional[TrussSpecDecMode]
    max_draft_len: Optional[int]

    @validator("max_beam_width")
    def check_max_beam_width(cls, v: int):
        if isinstance(v, int):
            if v != 1:
                raise ValueError(
                    "max_beam_width greater than 1 is not currently supported"
                )
        return v


class TrussTRTLLMRuntimeConfiguration(BaseModel):
    kv_cache_free_gpu_mem_fraction: float = 0.9
    enable_chunked_context: bool = False
    num_draft_tokens: Optional[int]
    batch_scheduler_policy: TrussTRTLLMBatchSchedulerPolicy = (
        TrussTRTLLMBatchSchedulerPolicy.GUARANTEED_NO_EVICT
    )


class TRTLLMConfiguration(BaseModel):
    runtime: Optional[TrussTRTLLMRuntimeConfiguration] = None
    build: Optional[TrussTRTLLMBuildConfiguration] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_minimum_required_configuration()
        self._validate_kv_cache_flags()
        if self.build.checkpoint_repository.source == CheckpointSource.HF:
            self._validate_hf_repo_id()
        self._validate_spec_dec()

    # In pydantic v2 this would be `@model_validator(mode="after")` and
    # the __init__ override can be removed.
    def _validate_minimum_required_configuration(self):
        if not self.build:
            raise ValueError("Build configuration must be provided")
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

    def _validate_spec_dec(self):
        spec_dec_configs = [
            self.build.speculative_decoding_mode,
            self.build.max_draft_len,
            self.runtime.num_draft_tokens,
        ]
        if any(spec_dec_configs):
            if not all(spec_dec_configs):
                raise ValueError(
                    "Speculative Decoding requires all of `trt_llm.build.speculative_decoding`, `trt_llm.build.max_draft_len`, and `trt_llm.runtime.num_draft_tokens` to be configured."
                )

    @property
    def requires_build(self):
        if self.build is not None:
            return True
        return False

    # TODO(Abu): Replace this with model_dump(json=True)
    # when pydantic v2 is used here
    def to_json_dict(self, verbose=True):
        return json.loads(self.json(exclude_unset=not verbose))
