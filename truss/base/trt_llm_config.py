from __future__ import annotations

import logging
import os
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, Literal, Optional

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import (
    BaseModel,
    PydanticDeprecatedSince20,
    StringConstraints,
    model_validator,
    validator,
)

if TYPE_CHECKING:
    from truss.base.truss_config import TrussConfig

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
    webserver_default_route: Optional[
        Literal["/v1/embeddings", "/rerank", "/predict"]
    ] = None


class TrussTRTLLMLoraConfiguration(BaseModel):
    max_lora_rank: int = 64
    lora_target_modules: list[str] = []


class TrussTRTLLMBuildConfiguration(BaseModel):
    base_model: TrussTRTLLMModel = TrussTRTLLMModel.DECODER
    max_seq_len: Optional[int] = None
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
    lora_adapters: Optional[
        Dict[
            Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+$")],
            CheckpointRepository,
        ]
    ] = None
    lora_configuration: Optional[TrussTRTLLMLoraConfiguration] = None

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_kv_cache_flags()
        self._validate_speculator_config()

    def model_post_init(self, __context):
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
        return (self.speculator is not None) and (
            self.speculator.speculative_decoding_mode == TrussSpecDecMode.DRAFT_EXTERNAL
        )

    @property
    def uses_lora(self) -> bool:
        return self.lora_adapters is not None and len(self.lora_adapters) > 0

    def _bei_specfic_migration(self):
        """performs embedding specfic optimizations (no kv-cache, high batch size)"""
        if self.base_model == TrussTRTLLMModel.ENCODER:
            # Encoder specific settings
            if self.max_seq_len:
                logger.info(
                    f"Your setting of `build.max_seq_len={self.max_seq_len}` is not used for embedding models, "
                    "and only respected for SequenceClassification models. "
                    "Automatically inferred from the model repo config.json -> `max_position_embeddings`"
                )
            # delayed import, as it is not available in all environments [Briton]
            from truss.base.constants import BEI_REQUIRED_MAX_NUM_TOKENS

            if self.max_num_tokens < BEI_REQUIRED_MAX_NUM_TOKENS:
                if self.max_num_tokens != 8192:
                    # only warn if it is not the default value
                    logger.warning(
                        f"build.max_num_tokens={self.max_num_tokens}, upgrading to {BEI_REQUIRED_MAX_NUM_TOKENS}"
                    )
                self = self.model_copy(
                    update={"max_num_tokens": BEI_REQUIRED_MAX_NUM_TOKENS}
                )
            # set page_kv_cache and use_paged_context_fmha to false for encoder
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
            if self.max_batch_size and self.max_batch_size > 64:
                logger.warning(
                    "We recommend speculative decoding for lower load on your servers, e.g. with batch-sizes below 32"
                    "To get better auto-tuned kernels, we recommend capping the max_batch_size to a more reasonable number e.g. `max_batch_size=64` or `max_batch_size=32`"
                    "If you have high batch-sizes, speculative decoding may not be beneficial for total throughput."
                    "If you want to use speculative decoding on high load, tune the concurrency settings for more aggressive autoscaling on Baseten."
                )

            if not all(
                [
                    self.plugin_configuration.use_paged_context_fmha,
                    self.plugin_configuration.paged_kv_cache,
                ]
            ):
                raise ValueError(
                    "KV cache block reuse must be enabled for speculative decoding target model."
                )

            if self.uses_draft_external and self.speculator.build:
                logger.warning(
                    "Draft external mode is a advanced feature. If you encounter issues, feel free to contact us. You may also try lookahead decoding mode instead."
                )
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
            lade_num_draft_tokens = self.lade_max_draft_len(  # required
                self.lookahead_windows_size,
                self.lookahead_ngram_size,
                self.lookahead_verification_set_size,
            )
            if self.num_draft_tokens is None:
                self.num_draft_tokens = lade_num_draft_tokens
            if self.num_draft_tokens > 512:
                logger.warning(
                    f"Lookahead decoding mode generates up to {self.num_draft_tokens} speculative tokens per step and may have performance implications. "
                    "We recommend a simpler config, e.g. lookahead_windows_size=7, lookahead_ngram_size=5, lookahead_verification_set_size=3."
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


class VersionOverrides(BaseModel):
    engine_builder_version: Optional[str] = None
    bei_version: Optional[str] = None
    briton_version: Optional[str] = None


class TRTLLMConfiguration(BaseModel):
    runtime: TrussTRTLLMRuntimeConfiguration = TrussTRTLLMRuntimeConfiguration()
    build: TrussTRTLLMBuildConfiguration
    version_overrides: VersionOverrides = VersionOverrides()

    def model_post_init(self, __context):
        self.add_bei_default_route()
        self.chunked_context_fix()

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

    def chunked_context_fix(self: "TRTLLMConfiguration") -> "TRTLLMConfiguration":
        """check if there is an error wrt. runtime.enable_chunked_context"""
        if (
            self.runtime.enable_chunked_context
            and (self.build.base_model != TrussTRTLLMModel.ENCODER)
            and not (
                self.build.plugin_configuration.use_paged_context_fmha
                and self.build.plugin_configuration.paged_kv_cache
            )
        ):
            logger.warning(
                "If trt_llm.runtime.enable_chunked_context is True, then trt_llm.build.plugin_configuration.use_paged_context_fmha and trt_llm.build.plugin_configuration.paged_kv_cache should be True. "
                "Setting trt_llm.build.plugin_configuration.use_paged_context_fmha and trt_llm.build.plugin_configuration.paged_kv_cache to True."
            )
            self.build = self.build.model_copy(
                update={
                    "plugin_configuration": self.build.plugin_configuration.model_copy(
                        update={"use_paged_context_fmha": True, "paged_kv_cache": True}
                    )
                }
            )

        return self

    def add_bei_default_route(self):
        if (
            self.runtime.webserver_default_route is None
            and self.build.base_model == TrussTRTLLMModel.ENCODER
            and not ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION
        ):
            # attemp to set the best possible default route client side.
            try:
                from transformers import AutoConfig

                hf_cfg = AutoConfig.from_pretrained(
                    self.build.checkpoint_repository.repo,
                    revision=self.build.checkpoint_repository.revision,
                )
                # simple heuristic to set the default route
                is_sequence_classification = (
                    "ForSequenceClassification" in hf_cfg.architectures[0]
                )
                route = "/predict" if is_sequence_classification else "/v1/embeddings"
                self.runtime = self.runtime.model_copy(
                    update={"webserver_default_route": route}
                )
                logger.info(
                    f"Setting default route to {route} for your encoder, as the model is a "
                    + (
                        "SequenceClassification Model."
                        if is_sequence_classification
                        else "Embeddings model."
                    )
                )
            except Exception:
                # access error, or any other issue
                pass

    @property
    def requires_build(self):
        return self.build is not None


def trt_llm_validation(config: "TrussConfig") -> "TrussConfig":
    # Inline importing truss_config, to avoid cycle. This dependency is a bit sketchy,
    # but we don't want this trt specific code to live in `truss.base` and we also don't
    # want to move `Accelerator` out of the truss config module.
    from truss.base import constants, truss_config

    if config.trt_llm:
        if config.trt_llm.build.base_model != TrussTRTLLMModel.ENCODER:
            current_tags = config.model_metadata.get("tags", [])
            if (
                constants.OPENAI_COMPATIBLE_TAG in current_tags
                and constants.OPENAI_NON_COMPATIBLE_TAG in current_tags
            ):
                raise ValueError(
                    f"TRT-LLM models should have either model_metadata['tags'] = ['{constants.OPENAI_COMPATIBLE_TAG}'] or ['{constants.OPENAI_NON_COMPATIBLE_TAG}']. "
                    f"Your current tags are both {current_tags}, which is invalid. Please remove one of the tags."
                )
            elif not (
                constants.OPENAI_COMPATIBLE_TAG in current_tags
                or constants.OPENAI_NON_COMPATIBLE_TAG in current_tags
            ):
                # only check this in engine-builder for catching old truss pushes and force them adopt the new tag.
                message = f"""TRT-LLM models should have model_metadata['tags'] = ['{constants.OPENAI_COMPATIBLE_TAG}'] (or ['{constants.OPENAI_NON_COMPATIBLE_TAG}']).
                     Your current tags are {current_tags}, which is has neither option. We require a active choice to be made.
                     For making the model compatible with OpenAI clients, we require to add the following to your config.
                     ```yaml
                     model_metadata:
                     tags:
                     - {constants.OPENAI_COMPATIBLE_TAG}
                     # for legacy behavior set above line to the following, which will break OpenAI compatibility explicitly.
                     # This was the old default behaviour if you used Baseten before March 19th 2025 or truss<=0.9.68
                     # `- {constants.OPENAI_NON_COMPATIBLE_TAG}`
                     ```
                     """
                if ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION:
                    raise ValueError(message)
                else:
                    logger.warning(message)
            elif constants.OPENAI_NON_COMPATIBLE_TAG in current_tags:
                logger.warning(
                    f"Model is marked as {constants.OPENAI_NON_COMPATIBLE_TAG}. This model will not be compatible with OpenAI clients directly. "
                    f"This is the deprecated legacy behavior, please update the tag to {constants.OPENAI_COMPATIBLE_TAG}."
                )

        if (
            config.trt_llm.build.quantization_type
            is TrussTRTLLMQuantizationType.WEIGHTS_ONLY_INT8
            and config.resources.accelerator.accelerator
            is truss_config.Accelerator.A100
        ):
            logger.warning(
                "Weight only int8 quantization on A100 accelerators is not recommended."
            )
        if config.resources.accelerator.accelerator in [
            truss_config.Accelerator.T4,
            truss_config.Accelerator.V100,
        ]:
            raise ValueError(
                "TRT-LLM is not supported on CUDA_COMPUTE_75 (T4) and CUDA_COMPUTE_70 (V100) GPUs"
                "the lowest supported CUDA compute capability is CUDA_COMPUTE_80 (A100) or A10G (CUDA_COMPUTE_86)"
            )
        elif config.trt_llm.build.quantization_type in [
            TrussTRTLLMQuantizationType.FP8,
            TrussTRTLLMQuantizationType.FP8_KV,
        ] and config.resources.accelerator.accelerator in [
            truss_config.Accelerator.A10G,
            truss_config.Accelerator.A100,
            truss_config.Accelerator.A100_40GB,
        ]:
            raise ValueError(
                "FP8 quantization is only supported on L4, H100, H200 "
                "accelerators or newer (CUDA_COMPUTE>=89)"
            )
        tensor_parallel_count = config.trt_llm.build.tensor_parallel_count

        if tensor_parallel_count != config.resources.accelerator.count:
            raise ValueError(
                "Tensor parallelism and GPU count must be the same for TRT-LLM"
            )

    return config
