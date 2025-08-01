from __future__ import annotations

import logging
import os
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Dict, Literal, Optional, Union

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    PydanticDeprecatedSince20,
    RootModel,
    StringConstraints,
    Tag,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from truss.base.truss_config import TrussConfig

logger = logging.getLogger(__name__)
# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION = (
    os.environ.get("ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION", "False") == "True"
)
try:
    from truss.base import custom_types

    PydanticTrTBaseModel = custom_types.ConfigModel
except ImportError:
    # fallback for briton
    PydanticTrTBaseModel = BaseModel  # type: ignore[assignment,misc]


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


class InferenceStack(str, Enum):
    v1 = "v1"
    v2 = "v2"


class TrussTRTLLMQuantizationType(str, Enum):
    NO_QUANT = "no_quant"
    WEIGHTS_ONLY_INT8 = "weights_int8"
    WEIGHTS_KV_INT8 = "weights_kv_int8"
    WEIGHTS_ONLY_INT4 = "weights_int4"
    WEIGHTS_INT4_KV_INT8 = "weights_int4_kv_int8"
    SMOOTH_QUANT = "smooth_quant"
    FP8 = "fp8"
    FP8_KV = "fp8_kv"
    FP4 = "fp4"
    FP4_KV = "fp4_kv"


class TrussTRTLLMPluginConfiguration(PydanticTrTBaseModel):
    paged_kv_cache: bool = True
    use_paged_context_fmha: bool = True
    use_fp8_context_fmha: bool = False


class TrussTRTQuantizationConfiguration(PydanticTrTBaseModel):
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


class CheckpointRepository(PydanticTrTBaseModel):
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


class TrussTRTLLMRuntimeConfiguration(PydanticTrTBaseModel):
    kv_cache_free_gpu_mem_fraction: float = 0.9
    kv_cache_host_memory_bytes: Optional[Annotated[int, Field(strict=True, ge=1)]] = (
        None
    )
    enable_chunked_context: bool = True
    batch_scheduler_policy: TrussTRTLLMBatchSchedulerPolicy = (
        TrussTRTLLMBatchSchedulerPolicy.GUARANTEED_NO_EVICT
    )
    request_default_max_tokens: Optional[Annotated[int, Field(strict=True, ge=1)]] = (
        None
    )
    served_model_name: Optional[str] = None
    total_token_limit: int = 500000
    webserver_default_route: Optional[
        Literal["/v1/embeddings", "/rerank", "/predict"]
    ] = None


class TRTLLMRuntimeConfigurationV2(PydanticTrTBaseModel):
    max_seq_len: Optional[Annotated[int, Field(strict=True, ge=1, le=1048576)]] = None
    max_batch_size: Annotated[int, Field(strict=True, ge=1, le=2048)] = 256
    max_num_tokens: Annotated[int, Field(strict=True, gt=64, le=131072)] = 8192
    tensor_parallel_size: Annotated[int, Field(strict=True, ge=1)] = 1
    enable_chunked_prefill: bool = True
    served_model_name: Optional[str] = None
    patch_kwargs: Dict[str, Union[str, int, float, dict]] = Field(default_factory=dict)

    @field_validator("patch_kwargs", mode="after")
    @classmethod
    def validate_patch_kwargs(cls, config):
        if config:
            logger.warning(
                "trt_llm.runtime.patch_kwargs is a preview feature. "
                "Fields may change in the future."
            )
        forbidden_keys = ["build_config"] + list(cls.__fields__)
        for key in forbidden_keys:
            if key in config:
                logger.error(
                    f"runtime.config_kwargs contains the key '{key}'. This is already a field in the TRTLLMRuntimeConfigurationV2. "
                    "Please use the appropriate field in the TRTLLMRuntimeConfigurationV2."
                )
        return config


class TrussTRTLLMLoraConfiguration(PydanticTrTBaseModel):
    max_lora_rank: int = 64
    lora_target_modules: list[str] = []


class TrussTRTLLMBuildConfiguration(PydanticTrTBaseModel):
    base_model: TrussTRTLLMModel = TrussTRTLLMModel.DECODER
    max_seq_len: Optional[Annotated[int, Field(strict=True, ge=1, le=1048576)]] = None
    max_batch_size: Annotated[int, Field(strict=True, ge=1, le=2048)] = 256
    max_num_tokens: Annotated[int, Field(strict=True, gt=64, le=1048576)] = 8192
    max_beam_width: Annotated[int, Field(strict=True, ge=1, le=1)] = (
        1  # "max_beam_width greater than 1 is not currently supported"
    )
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
    tensor_parallel_count: Annotated[int, Field(strict=True, ge=1)] = 1
    pipeline_parallel_count: int = 1
    sequence_parallel_count: int = 1
    plugin_configuration: TrussTRTLLMPluginConfiguration = (
        TrussTRTLLMPluginConfiguration()
    )
    num_builder_gpus: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    speculator: Optional[TrussSpeculatorConfiguration] = None
    lora_adapters: Optional[
        Dict[
            Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_\-\.:]+$")],
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


class TrussSpeculatorConfiguration(PydanticTrTBaseModel):
    speculative_decoding_mode: TrussSpecDecMode = TrussSpecDecMode.DRAFT_EXTERNAL
    num_draft_tokens: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    checkpoint_repository: Optional[CheckpointRepository] = None
    runtime: TrussTRTLLMRuntimeConfiguration = TrussTRTLLMRuntimeConfiguration()
    build: Optional[TrussTRTLLMBuildConfiguration] = None
    lookahead_windows_size: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    lookahead_ngram_size: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    lookahead_verification_set_size: Optional[
        Annotated[int, Field(strict=True, ge=1)]
    ] = None
    enable_b10_lookahead: Optional[bool] = False

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
        else:
            if self.enable_b10_lookahead:
                logger.warning(
                    "enable_b10_lookahead requires in `speculative_decoding_mode=LOOKAHEAD_DECODING`. "
                    "Please enable both flags to use the Baseten lookahead algorithm."
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


class VersionsOverrides(PydanticTrTBaseModel):
    # If an override is specified, it takes precedence over the backend's current
    # default version. The version is used to create a full image ref and should look
    # like a semver, e.g. for the briton the version `0.17.0-fd30ac1` could be specified
    # here and the backend creates the full image tag like
    # `baseten/briton-server:v0.17.0-fd30ac1`.
    engine_builder_version: Optional[str] = None
    briton_version: Optional[str] = None
    bei_version: Optional[str] = None
    v2_llm_version: Optional[str] = None

    @model_validator(mode="before")
    def version_must_start_with_number(cls, data):
        for field in ["engine_builder_version", "briton_version", "bei_version"]:
            v = data.get(field)
            if v is not None and (not v or not v[0].isdigit()):
                raise ValueError(f"{field} must start with a number")
        return data


class ImageVersions(PydanticTrTBaseModel):
    # Required versions for patching truss config during docker build setup.
    # The schema of this model must be such that it can parse the values serialized
    # from the backend. The inserted values are full image references, resolved using
    # backend defaults and `ImageVersionsOverrides` from the pushed config.
    bei_image: str
    briton_image: str
    v2_llm_image: str = (
        "baseten/dynamo-cache-aware-routing:trtllm-gpu-ea9f7cb-725b8f2-eff071c4a5"
    )


class TRTLLMConfigurationV1(PydanticTrTBaseModel):
    build: TrussTRTLLMBuildConfiguration
    inference_stack: Literal[InferenceStack.v1] = InferenceStack.v1
    runtime: TrussTRTLLMRuntimeConfiguration = TrussTRTLLMRuntimeConfiguration()

    # If versions are not set, the baseten backend will insert current defaults.
    version_overrides: VersionsOverrides = VersionsOverrides()

    def model_post_init(self, __context):
        """Post-initialization validation and adjustments."""
        if (
            self.runtime.enable_chunked_context
            and (self.build.base_model != TrussTRTLLMModel.ENCODER)
            and not (
                self.build.plugin_configuration.use_paged_context_fmha
                and self.build.plugin_configuration.paged_kv_cache
            )
        ):
            raise ValueError(
                "If trt_llm.runtime.enable_chunked_context is True, then trt_llm.build.plugin_configuration.use_paged_context_fmha and trt_llm.build.plugin_configuration.paged_kv_cache need to be True. "
            )

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


class TRTLLMConfigurationV2(PydanticTrTBaseModel):
    inference_stack: Literal[InferenceStack.v2] = InferenceStack.v2
    build: TrussTRTLLMBuildConfiguration
    runtime: TRTLLMRuntimeConfigurationV2
    version_overrides: VersionsOverrides = VersionsOverrides()

    @model_validator(mode="after")
    def validate_inference_stack_v2(self: "TRTLLMConfigurationV2", context):
        """Validate that the build configuration is compatible with the v2 inference stack."""
        allowed_modify_fields = [
            "checkpoint_repository",
            "quantization_type",
            "quantization_config",
        ]

        build_settings = self.build.model_dump(exclude_unset=True)
        build_settings_reference = TrussTRTLLMBuildConfiguration(
            checkpoint_repository=CheckpointRepository(
                source=CheckpointSource.HF, repo="michael/any", revision=None
            ),
            quantization_type=TrussTRTLLMQuantizationType.NO_QUANT,
            quantization_config=TrussTRTQuantizationConfiguration(),
        ).model_dump(exclude_unset=False)
        for field in build_settings:
            if (
                field not in allowed_modify_fields
                and build_settings[field] != build_settings_reference[field]
            ):
                raise ValueError(
                    f"Field trt_llm.build.{field} is not allowed to be set when using v2 inference stack. "
                )

        return self


def get_discriminator_value(v) -> str:
    if isinstance(v, dict) and v.get("inference_stack"):
        inference_stack = v["inference_stack"]
        if isinstance(inference_stack, InferenceStack):
            return inference_stack.value
        elif isinstance(inference_stack, str):
            return InferenceStack(inference_stack).value
    elif hasattr(v, "inference_stack"):
        inference_stack = v.inference_stack
        if isinstance(inference_stack, InferenceStack):
            return inference_stack.value
        elif isinstance(inference_stack, str):
            return InferenceStack(inference_stack).value
    elif isinstance(v, dict):
        return InferenceStack.v1.value
    raise ValueError(
        f"Invalid value for discriminator: {v}. Expected a dict with 'inference_stack' key."
    )


class TRTLLMConfiguration(RootModel):
    root: Annotated[
        Union[
            Annotated[TRTLLMConfigurationV1, Tag("v1")],
            Annotated[TRTLLMConfigurationV2, Tag("v2")],
        ],
        Discriminator(get_discriminator_value),
    ]

    @property
    def inference_stack(self) -> InferenceStack:
        """Returns the inference stack of the configuration."""
        return self.root.inference_stack

    @property
    def build(self) -> TrussTRTLLMBuildConfiguration:
        """Returns the build configuration of the TRT-LLM."""
        return self.root.build

    @property
    def runtime(
        self,
    ) -> Union[TrussTRTLLMRuntimeConfiguration, TRTLLMRuntimeConfigurationV2]:
        """Returns the runtime configuration of the TRT-LLM."""
        return self.root.runtime

    @property
    def version_overrides(self) -> VersionsOverrides:
        """Returns the version overrides for the TRT-LLM configuration."""
        return self.root.version_overrides

    @property
    def requires_build(self):
        """depreacted, always true."""
        return True


def trt_llm_validation(config: "TrussConfig") -> "TrussConfig":
    # Inline importing truss_config, to avoid cycle. This dependency is a bit sketchy,
    # but we don't want this trt specific code to live in `truss.base` and we also don't
    # want to move `Accelerator` out of the truss config module.
    if not config.trt_llm:
        return config
    elif isinstance(config.trt_llm.root, TRTLLMConfigurationV1):
        trt_llm_common_validation(config)
        return trt_llm_validation_v1(config)
    elif isinstance(config.trt_llm.root, TRTLLMConfigurationV2):
        # v2 inference stack does not require any validation, as it is fully compatible with the truss config.
        # It is only used for the new inference stack, which is fully compatible with the old one.
        trt_llm_common_validation(config)
        return trt_llm_validation_v2(config)
    else:
        raise ValueError(
            f"Unknown inference stack {config.trt_llm.inference_stack}. "
            "Please use either InferenceStack.v1 or InferenceStack.v2."
        )


def trt_llm_common_validation(config: "TrussConfig"):
    from truss.base import truss_config

    assert config.trt_llm, "TRT-LLM configuration is required for TRT-LLM models"
    trt_llm_config: TRTLLMConfigurationV1 | TRTLLMConfigurationV2 = config.trt_llm.root
    if (
        trt_llm_config.build.quantization_type
        is TrussTRTLLMQuantizationType.WEIGHTS_ONLY_INT8
        and config.resources.accelerator.accelerator is truss_config.Accelerator.A100
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
    elif trt_llm_config.build.quantization_type in [
        TrussTRTLLMQuantizationType.FP8,
        TrussTRTLLMQuantizationType.FP8_KV,
        TrussTRTLLMQuantizationType.FP4,
    ] and config.resources.accelerator.accelerator in [
        truss_config.Accelerator.A10G,
        truss_config.Accelerator.A100,
        truss_config.Accelerator.A100_40GB,
    ]:
        raise ValueError(
            "FP8 quantization is only supported on L4, H100, H200 "
            "accelerators or newer (CUDA_COMPUTE>=89)"
        )
    elif trt_llm_config.build.quantization_type in [
        TrussTRTLLMQuantizationType.FP4
    ] and config.resources.accelerator.accelerator in [
        truss_config.Accelerator.H100,
        truss_config.Accelerator.L4,
        truss_config.Accelerator.A100_40GB,
    ]:
        raise ValueError(
            "FP4 quantization is only supported on B200 / Blackwell "
            "accelerators or newer (CUDA_COMPUTE>=100)"
        )


def trt_llm_validation_v2(config: "TrussConfig") -> "TrussConfig":
    assert config.trt_llm, "TRT-LLM configuration is required for TRT-LLM models"
    assert isinstance(config.trt_llm.root, TRTLLMConfigurationV2), (
        "TRT-LLM configuration must be of type TRTLLMConfigurationV2 for v2 inference stack"
    )
    return config


def trt_llm_validation_v1(config: "TrussConfig") -> "TrussConfig":
    from truss.base import constants

    assert config.trt_llm, "TRT-LLM configuration is required for TRT-LLM models"
    assert isinstance(config.trt_llm.root, TRTLLMConfigurationV1), (
        "TRT-LLM configuration must be of type TRTLLMConfigurationV1 for v1 inference stack"
    )

    trt_llm_config_v1: "TRTLLMConfigurationV1" = config.trt_llm.root
    if trt_llm_config_v1.build.base_model != TrussTRTLLMModel.ENCODER:
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

    world_size = (
        trt_llm_config_v1.build.tensor_parallel_count
        * trt_llm_config_v1.build.pipeline_parallel_count
        * trt_llm_config_v1.build.sequence_parallel_count
    )

    if world_size != config.resources.accelerator.count:
        raise ValueError(
            "Tensor parallelism and GPU count must be the same for TRT-LLM"
            f"You have set tensor_parallel_count={trt_llm_config_v1.build.tensor_parallel_count}, "
            f"pipeline_parallel_count={trt_llm_config_v1.build.pipeline_parallel_count}, "
            f"sequence_parallel_count={trt_llm_config_v1.build.sequence_parallel_count} "
            f"== world_size->{world_size} "
            f"and accelerator.count={config.resources.accelerator.count}. "
        )

    return config


TRTLLMConfigurationV2(
    build=TrussTRTLLMBuildConfiguration(
        checkpoint_repository=CheckpointRepository(
            source=CheckpointSource.HF, repo="michael/any", revision=None
        ),
        quantization_type=TrussTRTLLMQuantizationType.NO_QUANT,
    ),
    runtime=TRTLLMRuntimeConfigurationV2(
        max_seq_len=2048,
        max_batch_size=256,
        max_num_tokens=8192,
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        config_kwargs={},
    ),
)
