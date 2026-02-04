from __future__ import annotations

import functools
import logging
import os
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, Literal, Optional, Union

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import (
    AliasChoices,
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
# Suppress Pydantic V1 warnings, because  we have to use it for backwards compat.
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
    # Causal models for embedding, but also does have basic Bert support.
    # Worlds fastest runtime for qwen3-8b (causal embedding models) and other causal embedding models.
    # also very fast for qwen, mistral, llama3 causal models for SequenceClassification
    ENCODER = "encoder"
    # BERT-based encoder models for non-causal tasks such as text classification, reranking, embeddings etc.
    # the encoder_bert setting will specfically optimize for thoughput and cold-start latency of small models (<4B parameters)
    # supports also splade and colbert style models or ModernBert.
    ENCODER_BERT = "encoder_bert"
    # Decoder will launch the backend that is optimized for decoder only models such as LLama3ForCausalLM, Qwen3MoeForCausalLM etc.
    DECODER = "decoder"
    # a ERROR will be raised if you push one of the below models. Don't use
    PALMYRA = "palmyra"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    # deprecated workflow
    WHISPER = "whisper"


class InferenceStack(str, Enum):
    # V1: use for small models, dense models. MoEs are not supported. Super fast speculation e.g. for code edit models
    # V1: used for BEI models (causal embedding and classifer models) and BEI-Bert (non-causal small embedding models)
    # All embedding runtimes are running without tensor parallelism (optimized for latency + thoughput.)
    v1 = "v1"
    # V2: Use for all MoE models like Qwen3Moe, DeepSeek, Kimi, GLM4.
    # use for multi node setups.
    v2 = "v2"


class TrussTRTLLMQuantizationType(str, Enum):
    # no_quant means fp16 or bf16. It will use the precision prefered in the config.json and available on the GPU.
    NO_QUANT = "no_quant"
    # the below WEIGHTS_* and SMOOTH_QUANT are legacy quantization types
    # and are raising a error on push. Don't use.
    WEIGHTS_ONLY_INT8 = "weights_int8"
    WEIGHTS_KV_INT8 = "weights_kv_int8"
    WEIGHTS_ONLY_INT4 = "weights_int4"
    WEIGHTS_INT4_KV_INT8 = "weights_int4_kv_int8"
    SMOOTH_QUANT = "smooth_quant"
    # FP8 weights + 16 bit kv cache
    FP8 = "fp8"
    # FP8 + fp8 kv cache quantization (faster attention when used with fp8 context fmha, required for fp8 ctx fmha)!
    # not usable for asymmetric model with bias=True e.g. qwen2.5 models
    FP8_KV = "fp8_kv"
    # fp4 with 16 bit kv cache
    FP4 = "fp4"
    # fp4 with fp8 kv cache quantization
    FP4_KV = "fp4_kv"
    # fp4, but only mlp layers are in fp4, rest is 16 bit , also 16 bit kv cache
    FP4_MLP_ONLY = "fp4_mlp_only"


class TrussTRTLLMPluginConfiguration(PydanticTrTBaseModel):
    # strongly recommend to always have on. Do not set to false.
    paged_kv_cache: bool = True
    # strongly recommend to always have on. Do not set to false.
    use_paged_context_fmha: bool = True
    # recommend to have one when using fp8 quantization of kv cache
    # AUTO-ENABLED: Has no effect.
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
    calib_dataset: str = "abisee/cnn_dailymail"
    calib_max_seq_length: int = 1536

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
    # AZURE, GCS and S3 use the `model_cache` to download the checkpoint.
    # The repo should be the bucket name + path within the bucket.
    # e.g. s3://my-bucket/path/to/checkpoint, where checkpoint contains config.json, model.safetensors etc.
    # safetensors format strongly needed, as most operations will fail with pytorch.bin format.
    GCS = "GCS"
    S3 = "S3"
    AZURE = "AZURE"
    # LOCAL is not supported, will raise error, do not document.
    LOCAL = "LOCAL"
    # REMOTE_URL is useful when the checkpoint lives on remote storage accessible via HTTP (e.g a presigned URL)
    # as tar.gzip file.
    REMOTE_URL = "REMOTE_URL"
    # if deploying from a baseten training job
    # the repo with be the training job and the revision the revision of the training run.
    BASETEN_TRAINING = "BASETEN_TRAINING"


class CheckpointRepository(PydanticTrTBaseModel):
    source: CheckpointSource
    # repo and revision semantics depend on source. See above.
    repo: str
    revision: Optional[str] = None
    # secret from baseten secrets e.g. `hf_access_token`.
    # the secret in baseten Secret UI contains the actual token or credentials., this is just the file name.
    runtime_secret_name: str = "hf_access_token"

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
    # max utlilization: is recommend if you are serving customer request where you can't control the requested number of max_tokens.
    # As it will schedule the requests without looking at the available.
    # may need to stop/pause multiple requests if memory gets full (not recommened )
    MAX_UTILIZATION = "max_utilization"
    # default: will guarantee the scheduling of a request with the number of tokens that the user requested.
    # may queue requests if not enough memory is available.
    GUARANTEED_NO_EVICT = "guaranteed_no_evict"


class TrussSpecDecMode(str, Enum):
    # draft external is deprecated, use LOOKAHEAD_DECODING instead. No longer documented.
    DRAFT_EXTERNAL = "DRAFT_TOKENS_EXTERNAL"
    # lookahead decoding uses internal draft generation with briton. Recommended.
    # It is roughtly based on n-gram speculation built from the ground up.
    # Its the world fastest speculation method for trt_llm models e.g. for code edits where n-gram like speculation
    # works well
    LOOKAHEAD_DECODING = "LOOKAHEAD_DECODING"


class TrussTRTLLMRuntimeConfiguration(PydanticTrTBaseModel):
    # how much context length to support
    kv_cache_free_gpu_mem_fraction: float = 0.9
    # how much memory to reserve on host (CPU) for kv-cache (in bytes)
    # set to a high value to enable kv-cache offload to host memory
    kv_cache_host_memory_bytes: Optional[Annotated[int, Field(strict=True, ge=1)]] = (
        None
    )
    # wheter to
    enable_chunked_context: bool = True
    batch_scheduler_policy: TrussTRTLLMBatchSchedulerPolicy = (
        TrussTRTLLMBatchSchedulerPolicy.GUARANTEED_NO_EVICT
    )
    request_default_max_tokens: Optional[Annotated[int, Field(strict=True, ge=1)]] = (
        None
    )
    # only for generative models (e.g. decoder models)
    served_model_name: Optional[str] = None
    # how many tokens get scheduled at once to the C++ engine.
    # only applicable for generative models (e.g. decoder models)
    total_token_limit: int = 500000
    # only for embedding models (e.g. encoder models and encoder_bert models)
    webserver_default_route: Optional[
        Literal["/v1/embeddings", "/rerank", "/predict"]
    ] = None


class TRTLLMRuntimeConfigurationV2(PydanticTrTBaseModel):
    max_seq_len: Optional[Annotated[int, Field(strict=True, ge=1, le=1048576)]] = None
    # how many requests can be batched together in one forward pass
    max_batch_size: Annotated[int, Field(strict=True, ge=1, le=2048)] = 256
    # how many tokens can be gbatched together in one forward pass
    max_num_tokens: Annotated[int, Field(strict=True, gt=64, le=131072)] = 8192
    tensor_parallel_size: Annotated[int, Field(strict=True, ge=1)] = 1
    # whether to enable chunked prefill for generative models (decoder models)
    enable_chunked_prefill: bool = True
    # only for generative models (e.g. decoder models), name in the json response
    served_model_name: Optional[str] = None
    # only for V2 inference stack, advanced use.
    patch_kwargs: Dict[str, Union[str, int, float, dict, list, None]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("patch_kwargs", "gated_features"),
    )

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
    # for BEI/encoder and for generative models without chunked prefill:
    # This will limit the max context length of input (+output token length for generative models)
    max_num_tokens: Annotated[int, Field(strict=True, gt=64, le=1048576)] = 8192
    # do not document, only 1 is allowed.
    max_beam_width: Annotated[int, Field(strict=True, ge=1, le=1)] = (
        1  # "max_beam_width greater than 1 is not currently supported"
    )
    max_prompt_embedding_table_size: int = 0
    checkpoint_repository: CheckpointRepository
    gather_all_token_logits: bool = False
    # if you want to ignore the dtype of the model you loaded.
    # recommend to not use unless you get a error during the build (model failing with compile error)
    strongly_typed: bool = False
    quantization_type: TrussTRTLLMQuantizationType = (
        TrussTRTLLMQuantizationType.NO_QUANT
    )
    quantization_config: TrussTRTQuantizationConfiguration = (
        TrussTRTQuantizationConfiguration()
    )
    # number of GPUs to use for tensor parallelism
    # only for decoder models with inference_stack v1 / v2
    tensor_parallel_count: Annotated[int, Field(strict=True, ge=1)] = 1
    # pipeline parallel count not allowed.
    pipeline_parallel_count: int = 1
    moe_expert_parallel_option: int = -1
    sequence_parallel_count: int = 1
    plugin_configuration: TrussTRTLLMPluginConfiguration = (
        TrussTRTLLMPluginConfiguration()
    )
    # if you are running quantization on the same GPU as for inference, you might run out of GPU memory.
    # if you are running out of GPU memory, set to a higher number than the deployment (aka higher than tp)
    # if you are running out of CPU memory, do not adjust this number and add more CPU memory in the resource section.
    num_builder_gpus: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    # config for lookahead speculative decoding
    speculator: Optional[TrussSpeculatorConfiguration] = None
    # for v1 decoder models only, a ahead of time known set of lora adapters.
    # the name will be used as name of the openai client `model` key
    lora_adapters: Optional[
        Dict[
            Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_\-\.:]+$")],
            CheckpointRepository,
        ]
    ] = None
    lora_configuration: Optional[TrussTRTLLMLoraConfiguration] = None
    # for v2, skip the build step and use a engine that you e.g. provider otherwise
    # e.g. via model_cache.
    skip_build_result: bool = False

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

    def validate_not_external_draft(self):
        if self.uses_draft_external:
            raise ValueError(
                """
external draft speculation is discontinued in favor of LOOKAHEAD_DECODING, which yields better performance.
Alternatively, switch to `inference_stack: v2` and enabled more sophisticated speculation methods.
As last resort, you may build with the following overrides:

```
trt_llm:
  version_overrides:
    briton_version: 0.20.0_v0.1.5rc1
    engine_builder_version: 0.20.0.post8.dev1
```

and install this truss version:
```
pip install truss==0.10.8
```

"""
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
                logger.warning(
                    "Compling `encoder` with a kv-cache dtype is a alpha feature. This may fail. "
                    f"You selected build.quantization_type {self.quantization_type}"
                )

    def _validate_kv_cache_flags(self):
        if not self.plugin_configuration.paged_kv_cache and (
            self.plugin_configuration.use_paged_context_fmha
        ):
            raise ValueError(
                "Using paged context fmha requires requires paged kv cache"
            )

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

            self.validate_not_external_draft()

    @model_validator(mode="after")
    def validate_moe_parallelism_valid(self):
        if self.base_model == TrussTRTLLMModel.DECODER:
            if self.moe_expert_parallel_option != -1:
                if self.moe_expert_parallel_option > self.tensor_parallel_count:
                    raise ValueError(
                        f"moe_expert_parallel_option {self.moe_expert_parallel_option} cannot be greater than tensor_parallel_count {self.tensor_parallel_count}"
                    )
                if self.moe_expert_parallel_option <= 0:
                    raise ValueError(
                        f"moe_expert_parallel_option {self.moe_expert_parallel_option} must be positive or -1"
                    )
                if self.tensor_parallel_count % self.moe_expert_parallel_option != 0:
                    logger.warning(
                        f"tensor_parallel_count {self.tensor_parallel_count} is not divisible by moe_expert_parallel_option {self.moe_expert_parallel_option}. This may lead to suboptimal performance."
                    )
        return self

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
    # recommened according to lookahead paper
    # (8,3,3), (5,5,5)
    # setting it to (*,1,1) e.g. (8,1,1) will allow dynamic speculation, if enable_b10_lookahead is enabled.
    # in this case (4,1,1), (8,1,1) and (32,1,1) are recommended.

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
    # INTERNAL
    bei_image: str
    beibert_image: str = (
        "baseten/bei_bert:1.8.6"  # once wired up in core-product, this can be removed
    )
    briton_image: str
    v2_llm_image: str


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
            and (
                self.build.base_model
                not in (TrussTRTLLMModel.ENCODER, TrussTRTLLMModel.ENCODER_BERT)
            )
            and not (
                self.build.plugin_configuration.use_paged_context_fmha
                and self.build.plugin_configuration.paged_kv_cache
            )
        ):
            raise ValueError(
                "If trt_llm.runtime.enable_chunked_context is True, then trt_llm.build.plugin_configuration.use_paged_context_fmha and trt_llm.build.plugin_configuration.paged_kv_cache need to be True. "
            )
        hf_cfg = get_hf_config(
            repo=self.build.checkpoint_repository.repo,
            revision=self.build.checkpoint_repository.revision,
        )

        if (
            self.runtime.webserver_default_route is None
            and self.build.base_model
            in (TrussTRTLLMModel.ENCODER, TrussTRTLLMModel.ENCODER_BERT)
            and not ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION
        ):
            if hf_cfg is not None:
                arch = hf_cfg.architectures[0]
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
        if hf_cfg is not None:
            arch = hf_cfg.architectures[0]
            if (
                "bert" in arch.lower()
                and self.build.base_model != TrussTRTLLMModel.ENCODER_BERT
            ):
                logger.warning(
                    f"Your model architecture {arch} indicates a BERT-like based model. "
                    f"Consider setting `trt_llm.build.base_model` to `encoder_bert` for better performance and compatibility."
                )
                if self.build.base_model == TrussTRTLLMModel.DECODER:
                    logger.error(
                        f"Your model architecture {arch} indicates a BERT-like based model. "
                        f"but you set `trt_llm.build.base_model` to `decoder`. "
                        f"Please set it to `encoder_bert`."
                    )
            if (
                "ForCausalLM" in arch
                and self.build.base_model != TrussTRTLLMModel.DECODER
            ):
                logger.warning(
                    f"Your model architecture {arch} indicates a CausalLM based model. "
                    f"Consider setting `trt_llm.build.base_model` to `decoder` for better performance and compatibility."
                )
                if self.build.base_model in (
                    TrussTRTLLMModel.ENCODER,
                    TrussTRTLLMModel.ENCODER_BERT,
                ):
                    logger.error(
                        f"Your model architecture {arch} indicates a CausalLM based model. "
                        f"but you set `trt_llm.build.base_model` to `encoder` or `encoder_bert`. "
                        " Deploy it as `decoder` model instead, if you want to use it for text-generation. (most likley this is what you do)"
                        " In the rare event you want to use it for Sequence classification via the first logit only:"
                        " Please covert the CausalLM head using this script: "
                        " https://github.com/michaelfeil/infinity/blob/c030718f3bf163237caa8898ee59cd53ba213124/docs/lm_head_to_classifier/convert_lm.py"
                    )
            if (
                "ForSequenceClassification" in arch
                and self.build.base_model == TrussTRTLLMModel.DECODER
            ):
                logger.error(
                    f"Your model architecture {arch} indicates a SequenceClassification based model, "
                    f"but you set `trt_llm.build.base_model` to `decoder`. "
                    f"Please set it to `encoder` or `encoder_bert`."
                )


@functools.lru_cache(maxsize=4)
def get_hf_config(repo: str, revision: Optional[str]) -> Optional[Any]:
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(
            repo, revision=revision, trust_remote_code=False
        )
        return hf_cfg
    except Exception:
        return None


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
            "skip_build_result",
            "num_builder_gpus",
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
    base_model = (
        trt_llm_config.build.base_model
        if hasattr(trt_llm_config.build, "base_model")
        else None
    )
    if (
        trt_llm_config.build.quantization_type
        is TrussTRTLLMQuantizationType.WEIGHTS_ONLY_INT8
        and config.resources.accelerator.accelerator is truss_config.Accelerator.A100
    ):
        logger.warning(
            "Weight only int8 quantization on A100 accelerators is not recommended."
        )
    if base_model in [
        TrussTRTLLMModel.PALMYRA,
        TrussTRTLLMModel.QWEN,
        TrussTRTLLMModel.LLAMA,
        TrussTRTLLMModel.MISTRAL,
        TrussTRTLLMModel.DEEPSEEK,
    ]:
        raise ValueError(
            f"{base_model} has been renamed to `decoder` in trt_llm.build.base_model. "
            " The decoder base_model now automatically detects the model architecture (e.g. Qwen, Llama, Mistral, etc.) from the checkpoint repository. "
            " The functionality remains the same, only the name has changed to better reflect the usage of the base_model field."
        )
    if base_model == TrussTRTLLMModel.WHISPER:
        raise ValueError(
            "Whisper models has been refactored to a Chain's version. "
            " Please send us a message on Slack or Support if you want to deploy a Whisper model with truss."
        )
    if config.resources.accelerator.accelerator in [
        truss_config.Accelerator.T4,
        truss_config.Accelerator.V100,
    ]:
        if base_model in [TrussTRTLLMModel.ENCODER_BERT]:
            # ENCODER_BERT runs fine on T4 + Bert backend.
            pass
        else:
            raise ValueError(
                "TRT-LLM is not supported on CUDA_COMPUTE_75 (T4) and CUDA_COMPUTE_70 (V100) GPUs. \n"
                "the lowest supported CUDA compute capability is CUDA_COMPUTE_80 (A100) or A10G (CUDA_COMPUTE_86)"
            )
    elif trt_llm_config.build.quantization_type in [
        TrussTRTLLMQuantizationType.FP8,
        TrussTRTLLMQuantizationType.FP8_KV,
        TrussTRTLLMQuantizationType.FP4,
        TrussTRTLLMQuantizationType.FP4_KV,
        TrussTRTLLMQuantizationType.FP4_MLP_ONLY,
    ] and config.resources.accelerator.accelerator in [
        truss_config.Accelerator.A10G,
        truss_config.Accelerator.A100,
        truss_config.Accelerator.A100_40GB,
        truss_config.Accelerator.T4,
        truss_config.Accelerator.V100,
    ]:
        raise ValueError(
            "FP8 quantization is only supported on L4, H100, H200, B200 "
            "accelerators or newer (CUDA_COMPUTE>=89)"
        )
    elif trt_llm_config.build.quantization_type in [
        TrussTRTLLMQuantizationType.FP4,
        TrussTRTLLMQuantizationType.FP4_KV,
        TrussTRTLLMQuantizationType.FP4_MLP_ONLY,
    ] and config.resources.accelerator.accelerator in [
        truss_config.Accelerator.H100,
        truss_config.Accelerator.H100_40GB,
        truss_config.Accelerator.H200,
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
    if trt_llm_config_v1.build.base_model not in [
        TrussTRTLLMModel.ENCODER,
        TrussTRTLLMModel.ENCODER_BERT,
    ]:
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
    if world_size not in [1, 2, 4, 8, 16, 32, 64, 128]:
        logger.warning(
            f"TRT-LLM world size {world_size} is unusual. Typical world sizes are powers of two, often 1 or [2,4,8]."
        )

    return config
