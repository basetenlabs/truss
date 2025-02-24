from __future__ import annotations

import json
import logging
import os
import warnings
from enum import Enum
from typing import Any, Optional

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import BaseModel, PydanticDeprecatedSince20, model_validator, validator

logger = logging.getLogger(__name__)
# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION = (
    os.environ.get("ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION", "False") == "True"
)


class TrussTRTLLMModel(str, Enum):
    ENCODER = "encoder"
    DECODER = "decoder"
    # auto migrated settings
    PALMYRA = "palmyra"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    # deprecated workflow
    WHISPER = "whisper"


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
    use_paged_context_fmha: bool = True
    use_fp8_context_fmha: bool = False


class TrussTRTQuantizationConfiguration(BaseModel):
    """Configuration for quantization of TRT models

    Args:
        calib_size (int, optional): Size of calibration dataset. Defaults to 1024.
            recommended to increase for production runs (e.g. 1536), or decrease e.g. to 256 for quick testing.
        calib_dataset (str, optional): Hugginface dataset to use for calibration. Defaults to 'cnn_dailymail'.
            uses split='train' and  quantized based on 'text' column.
        calib_max_seq_length (int, optional): Maximum sequence length for calibration. Defaults to 2048.
    """

    calib_size: int = 1024
    calib_dataset: str = "cnn_dailymail"
    calib_max_seq_length: int = 2048

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_cuda_friendly("calib_size")
        self.validate_cuda_friendly("calib_max_seq_length")

    def validate_cuda_friendly(self, key):
        value = getattr(self, key)
        if value < 64 or value > 16384:
            raise ValueError(f"{key} must be between 64 and 16384, but got {value}")
        elif value % 64 != 0:
            raise ValueError(f"{key} must be a multiple of 64, but got {value}")


class CheckpointSource(str, Enum):
    HF = "HF"
    GCS = "GCS"
    LOCAL = "LOCAL"
    # REMOTE_URL is useful when the checkpoint lives on remote storage accessible via HTTP (e.g a presigned URL)
    REMOTE_URL = "REMOTE_URL"


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
    DRAFT_EXTERNAL = "DRAFT_TOKENS_EXTERNAL"
    LOOKAHEAD_DECODING = "LOOKAHEAD_DECODING"


class TrussTRTLLMRuntimeConfiguration(BaseModel):
    kv_cache_free_gpu_mem_fraction: float = 0.9
    kv_cache_host_memory_bytes: Optional[int] = None
    enable_chunked_context: bool = True
    batch_scheduler_policy: TrussTRTLLMBatchSchedulerPolicy = (
        TrussTRTLLMBatchSchedulerPolicy.GUARANTEED_NO_EVICT
    )
    request_default_max_tokens: Optional[int] = None
    total_token_limit: int = 500000


class TrussTRTLLMBuildConfiguration(BaseModel):
    base_model: TrussTRTLLMModel = TrussTRTLLMModel.DECODER
    max_seq_len: int
    max_batch_size: int = 256
    max_num_tokens: int = 8192
    max_beam_width: int = 1
    max_prompt_embedding_table_size: int = 0
    checkpoint_repository: CheckpointRepository
    gather_all_token_logits: bool = False
    strongly_typed: bool = False
    quantization_type: TrussTRTLLMQuantizationType = (
        TrussTRTLLMQuantizationType.NO_QUANT
    )
    quantization_config: TrussTRTQuantizationConfiguration = (
        TrussTRTQuantizationConfiguration()
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

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_kv_cache_flags()
        self._validate_speculator_config()
        self._bei_specfic_migration()

    @validator("max_beam_width")
    def check_max_beam_width(cls, v: int):
        if isinstance(v, int):
            if v != 1:
                raise ValueError(
                    "max_beam_width greater than 1 is not currently supported"
                )
        return v

    @property
    def uses_lookahead_decoding(self) -> bool:
        return (
            self.speculator is not None
            and self.speculator.speculative_decoding_mode
            == TrussSpecDecMode.LOOKAHEAD_DECODING
        )

    @property
    def uses_draft_external(self) -> bool:
        return (
            self.speculator is not None
            and self.speculator.speculative_decoding_mode
            == TrussSpecDecMode.DRAFT_EXTERNAL
        )

    def _bei_specfic_migration(self):
        """performs embedding specfic optimizations (no kv-cache, high batch size)"""
        if self.base_model == TrussTRTLLMModel.ENCODER:
            # Encoder specific settings
            logger.info(
                f"Your setting of `build.max_seq_len={self.max_seq_len}` is not used and "
                "automatically inferred from the model repo config.json -> `max_position_embeddings`"
            )
            from truss.base.constants import BEI_REQUIRED_MAX_NUM_TOKENS

            if self.max_num_tokens < BEI_REQUIRED_MAX_NUM_TOKENS:
                logger.warning(
                    f"build.max_num_tokens={self.max_num_tokens}, upgrading to {BEI_REQUIRED_MAX_NUM_TOKENS}"
                )
                self.max_num_tokens = BEI_REQUIRED_MAX_NUM_TOKENS
            self.plugin_configuration.paged_kv_cache = False
            self.plugin_configuration.use_paged_context_fmha = False

            if "_kv" in self.quantization_type.value:
                raise ValueError(
                    "encoder does not have a kv-cache, therefore a kv specfic datatype is not valid"
                    f"you selected build.quantization_type {self.quantization_type}"
                )

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
        if (
            self.plugin_configuration.use_fp8_context_fmha
            and not self.quantization_type == TrussTRTLLMQuantizationType.FP8_KV
        ):
            raise ValueError("Using fp8 context fmha requires fp8 kv cache dtype")
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


class TrussSpeculatorConfiguration(BaseModel):
    speculative_decoding_mode: TrussSpecDecMode = TrussSpecDecMode.DRAFT_EXTERNAL
    num_draft_tokens: Optional[int] = None
    checkpoint_repository: Optional[CheckpointRepository] = None
    runtime: TrussTRTLLMRuntimeConfiguration = TrussTRTLLMRuntimeConfiguration()
    build: Optional[TrussTRTLLMBuildConfiguration] = None
    lookahead_windows_size: Optional[int] = None
    lookahead_ngram_size: Optional[int] = None
    lookahead_verification_set_size: Optional[int] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_checkpoint()
        self._validate_spec_dec_mode()

    def _assert_draft_tokens(self):
        if self.num_draft_tokens > 2048 or self.num_draft_tokens < 0:
            if self.speculative_decoding_mode == TrussSpecDecMode.LOOKAHEAD_DECODING:
                reason = (
                    f"This is automatically calculated value of lookahead_windows_size={self.lookahead_windows_size}, "
                    f" lookahead_ngram_size={self.lookahead_ngram_size}, lookahead_verification_set_size={self.lookahead_verification_set_size}. "
                    f"Please lower any of them."
                )
            else:
                reason = "You set this value under speculator.num_draft_tokens"
            raise ValueError(
                f"num_draft_tokens must be less than or equal to 2048. But you requested num_draft_tokens={self.num_draft_tokens}. {reason}"
            )

    @staticmethod
    def lade_max_draft_len(
        windows_size: int, ngram_size: int, verification_set_size: int
    ) -> int:
        """calculate the maximum number of tokens with baseten lookahead algorithm:  https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/lookahead#overview"""
        return (0 if (ngram_size == 1) else ngram_size - 2) + (
            windows_size - 1 + verification_set_size
        ) * (ngram_size - 1)

    def _validate_spec_dec_mode(self):
        if self.speculative_decoding_mode == TrussSpecDecMode.DRAFT_EXTERNAL:
            if not self.num_draft_tokens:
                raise ValueError(
                    "Draft external mode requires num_draft_tokens to be set."
                )
        elif self.speculative_decoding_mode == TrussSpecDecMode.LOOKAHEAD_DECODING:
            if not all(
                [
                    self.lookahead_windows_size,
                    self.lookahead_ngram_size,
                    self.lookahead_verification_set_size,
                ]
            ):
                raise ValueError(
                    f"Lookahead decoding mode requires lookahead_windows_size, lookahead_ngram_size, lookahead_verification_set_size to be set. you set: {self}"
                )
            lade_num_draft_tokens = self.lade_max_draft_len(
                self.lookahead_windows_size,
                self.lookahead_ngram_size,
                self.lookahead_verification_set_size,
            )
            if not ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION:
                if (
                    self.num_draft_tokens
                    and self.num_draft_tokens != lade_num_draft_tokens
                ):
                    raise ValueError(
                        f"num_draft_tokens is automatically calculated based on lookahead_windows_size, lookahead_ngram_size, lookahead_verification_set_size. "
                        f"Please remove num_draft_tokens or set it to exactly {lade_num_draft_tokens}. You set it to {self.num_draft_tokens}."
                    )
                self.num_draft_tokens = lade_num_draft_tokens
                if self.num_draft_tokens > 512:
                    logger.warning(
                        f"Lookahead decoding mode generates up to {self.num_draft_tokens} speculative tokens per step and may have performance implications. "
                        "We recommend a simpler config, e.g. lookahead_windows_size=7, lookahead_ngram_size=5, lookahead_verification_set_size=3."
                    )
            else:
                # server side on engine-builder
                if not self.num_draft_tokens:
                    raise ValueError(
                        "num_draft_tokens is required in lookahead decoding mode but not set"
                    )
                if (
                    self.num_draft_tokens < lade_num_draft_tokens
                ):  # check that it has at least the required tokens. That way, it could have even higher at request time.
                    raise ValueError(
                        "num_draft_tokens is less than the calculated value based on lookahead_windows_size, lookahead_ngram_size, lookahead_verification_set_size"
                    )

        self._assert_draft_tokens()

    def _validate_checkpoint(self):
        if self.speculative_decoding_mode == TrussSpecDecMode.DRAFT_EXTERNAL and not (
            bool(self.checkpoint_repository) ^ bool(self.build)
        ):
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
                if data.get("runtime"):
                    data.get("runtime").update(
                        {
                            k: v
                            for k, v in extra_runtime_fields.items()
                            if k not in data.get("runtime")
                        }
                    )
                else:
                    data.update(
                        {"runtime": {k: v for k, v in extra_runtime_fields.items()}}
                    )
            data.update({"build": valid_build_fields})
            return data
        return data

    @model_validator(mode="after")
    def after(self: "TRTLLMConfiguration") -> "TRTLLMConfiguration":
        # check if there is an error wrt. runtime.enable_chunked_context
        if (
            self.runtime.enable_chunked_context
            and (self.build.base_model != TrussTRTLLMModel.ENCODER)
            and not (
                self.build.plugin_configuration.use_paged_context_fmha
                and self.build.plugin_configuration.paged_kv_cache
            )
        ):
            if ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION:
                logger.warning(
                    "If trt_llm.runtime.enable_chunked_context is True, then trt_llm.build.plugin_configuration.use_paged_context_fmha and trt_llm.build.plugin_configuration.paged_kv_cache should be True. "
                    "Setting trt_llm.build.plugin_configuration.use_paged_context_fmha and trt_llm.build.plugin_configuration.paged_kv_cache to True."
                )
                self.build.plugin_configuration.use_paged_context_fmha = True
                self.build.plugin_configuration.paged_kv_cache = True
            else:
                raise ValueError(
                    "If runtime.enable_chunked_context is True, then build.plugin_configuration.use_paged_context_fmha and build.plugin_configuration.paged_kv_cache should be True"
                )

        return self

    @property
    def requires_build(self):
        return self.build is not None

    # TODO(Abu): Replace this with model_dump(json=True)
    # when pydantic v2 is used here
    def to_json_dict(self, verbose=True):
        return json.loads(self.json(exclude_unset=not verbose))
