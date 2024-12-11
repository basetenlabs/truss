from __future__ import annotations

import json
import logging
import warnings
from enum import Enum
from typing import Any, Optional

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import BaseModel, PydanticDeprecatedSince20, model_validator, validator

logger = logging.getLogger(__name__)
# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


class TrussTRTLLMModel(str, Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    WHISPER = "whisper"
    QWEN = "qwen"
    ENCODER = "encoder"


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
    revision: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.source == CheckpointSource.HF:
            self._validate_hf_repo_id()

    def _validate_hf_repo_id(self):
        try:
            validate_repo_id(self.repo)
        except HFValidationError as e:
            raise ValueError(
                f"HuggingFace repository validation failed: {str(e)}"
            ) from e


class TrussTRTLLMBatchSchedulerPolicy(str, Enum):
    MAX_UTILIZATION = "max_utilization"
    GUARANTEED_NO_EVICT = "guaranteed_no_evict"


class TrussSpecDecMode(str, Enum):
    DRAFT_EXTERNAL: str = "DRAFT_TOKENS_EXTERNAL"


class TrussTRTLLMRuntimeConfiguration(BaseModel):
    kv_cache_free_gpu_mem_fraction: float = 0.9
    enable_chunked_context: bool = False
    batch_scheduler_policy: TrussTRTLLMBatchSchedulerPolicy = (
        TrussTRTLLMBatchSchedulerPolicy.GUARANTEED_NO_EVICT
    )
    request_default_max_tokens: Optional[int] = None
    total_token_limit: int = 500000


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
    speculator: Optional[TrussSpeculatorConfiguration] = None

    class Config:
        extra = "forbid"

    @validator("max_beam_width")
    def check_max_beam_width(cls, v: int):
        if isinstance(v, int):
            if v != 1:
                raise ValueError(
                    "max_beam_width greater than 1 is not currently supported"
                )
        return v

    def _validate_kv_cache_flags(self):
        if not self.plugin_configuration.paged_kv_cache and (
            self.plugin_configuration.use_paged_context_fmha
            or self.plugin_configuration.use_fp8_context_fmha
        ):
            raise ValueError(
                "Using paged context fmha or fp8 context fmha requires requires paged kv cache"
            )
        if (
            self.plugin_configuration.use_fp8_context_fmha
            and not self.plugin_configuration.use_paged_context_fmha
        ):
            raise ValueError("Using fp8 context fmha requires paged context fmha")
        return self

    def _validate_speculator_config(self):
        if self.speculator:
            if self.base_model is TrussTRTLLMModel.WHISPER:
                raise ValueError("Speculative decoding for Whisper is not supported.")
            if not all(
                [
                    self.plugin_configuration.use_paged_context_fmha,
                    self.plugin_configuration.paged_kv_cache,
                ]
            ):
                raise ValueError(
                    "KV cache block reuse must be enabled for speculative decoding target model."
                )
            if self.speculator.build:
                if (
                    self.tensor_parallel_count
                    != self.speculator.build.tensor_parallel_count
                ):
                    raise ValueError(
                        "Speculative decoding requires the same tensor parallelism for target and draft models."
                    )

    @property
    def max_draft_len(self) -> Optional[int]:
        if self.speculator:
            return self.speculator.num_draft_tokens
        return None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_kv_cache_flags()
        self._validate_speculator_config()


class TrussSpeculatorConfiguration(BaseModel):
    speculative_decoding_mode: TrussSpecDecMode
    num_draft_tokens: int
    checkpoint_repository: Optional[CheckpointRepository] = None
    runtime: TrussTRTLLMRuntimeConfiguration = TrussTRTLLMRuntimeConfiguration()
    build: Optional[TrussTRTLLMBuildConfiguration] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_checkpoint()

    def _validate_checkpoint(self):
        if not (bool(self.checkpoint_repository) ^ bool(self.build)):
            raise ValueError(
                "Speculative decoding requires exactly one of checkpoint_repository or build to be configured."
            )

    @property
    def resolved_checkpoint_repository(self) -> CheckpointRepository:
        if self.build:
            return self.build.checkpoint_repository
        elif self.checkpoint_repository:
            return self.checkpoint_repository
        else:
            raise ValueError(
                "Speculative decoding requires exactly one of checkpoint_repository or build to be configured."
            )


class TRTLLMConfiguration(BaseModel):
    runtime: TrussTRTLLMRuntimeConfiguration = TrussTRTLLMRuntimeConfiguration()
    build: TrussTRTLLMBuildConfiguration

    @model_validator(mode="before")
    @classmethod
    def migrate_runtime_fields(cls, data: Any) -> Any:
        extra_runtime_fields = {}
        valid_build_fields = {}
        if isinstance(data.get("build"), dict):
            for key, value in data.get("build").items():
                if key in TrussTRTLLMBuildConfiguration.__annotations__:
                    valid_build_fields[key] = value
                else:
                    if key in TrussTRTLLMRuntimeConfiguration.__annotations__:
                        logger.warning(f"Found runtime.{key}: {value} in build config")
                        extra_runtime_fields[key] = value
            if extra_runtime_fields:
                logger.warning(
                    f"Found extra fields {list(extra_runtime_fields.keys())} in build configuration, unspecified runtime fields will be configured using these values."
                    " This configuration of deprecated fields is scheduled for removal, please upgrade to the latest truss version and update configs according to https://docs.baseten.co/performance/engine-builder-config."
                )
                data.get("runtime").update(
                    {
                        k: v
                        for k, v in extra_runtime_fields.items()
                        if k not in data.get("runtime")
                    }
                )
            data.update({"build": valid_build_fields})
            return data
        return data

    @property
    def requires_build(self):
        return self.build is not None

    # TODO(Abu): Replace this with model_dump(json=True)
    # when pydantic v2 is used here
    def to_json_dict(self, verbose=True):
        return json.loads(self.json(exclude_unset=not verbose))
