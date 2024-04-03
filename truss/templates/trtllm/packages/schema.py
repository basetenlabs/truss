from typing import Optional

import numpy as np
import tritonclient
import tritonclient.grpc.aio as grpcclient


class ModelInput:
    def __init__(
        self,
        prompt: str,
        request_id: int,
        max_tokens: int = 50,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        beam_width: int = 1,
        bad_words_list: Optional[list] = None,
        stop_words_list: Optional[list] = None,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        stream: bool = True,
        eos_token_id: int = None,  # type: ignore
    ) -> None:
        self.stream = stream
        self.request_id = request_id
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._beam_width = beam_width
        self._bad_words_list = [""] if bad_words_list is None else bad_words_list
        self._stop_words_list = [""] if stop_words_list is None else stop_words_list
        self._repetition_penalty = repetition_penalty
        self._eos_token_id = eos_token_id
        self._ignore_eos = ignore_eos
        # These variables are passed by OAI proxy but are unused
        # TODO(Abu): Add support for these
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

    def _prepare_grpc_tensor(
        self, name: str, input_data: np.ndarray
    ) -> grpcclient.InferInput:
        tensor = grpcclient.InferInput(
            name,
            input_data.shape,
            tritonclient.utils.np_to_triton_dtype(input_data.dtype),
        )
        tensor.set_data_from_numpy(input_data)
        return tensor

    def to_tensors(self):
        if self._eos_token_id is None and self._ignore_eos:
            raise ValueError("eos_token_id is required when ignore_eos is True")

        prompt_data = np.array([[self._prompt]], dtype=object)
        output_len_data = np.ones_like(prompt_data, dtype=np.uint32) * self._max_tokens
        bad_words_data = np.array([self._bad_words_list], dtype=object)
        stop_words_data = np.array([self._stop_words_list], dtype=object)
        stream_data = np.array([[self.stream]], dtype=bool)
        beam_width_data = np.array([[self._beam_width]], dtype=np.uint32)
        repetition_penalty_data = np.array(
            [[self._repetition_penalty]], dtype=np.float32
        )

        inputs = [
            self._prepare_grpc_tensor("text_input", prompt_data),
            self._prepare_grpc_tensor("max_tokens", output_len_data),
            self._prepare_grpc_tensor("bad_words", bad_words_data),
            self._prepare_grpc_tensor("stop_words", stop_words_data),
            self._prepare_grpc_tensor("stream", stream_data),
            self._prepare_grpc_tensor("beam_width", beam_width_data),
            self._prepare_grpc_tensor("repetition_penalty", repetition_penalty_data),
        ]

        if not self._ignore_eos:
            end_id_data = np.array([[self._eos_token_id]], dtype=np.uint32)
            inputs.append(self._prepare_grpc_tensor("end_id", end_id_data))

        return inputs


# The following are duplicated from the underlying base image.
# We list them as a comment for posterity:
#
# class TRTLLMModelArchitecture(Enum):
#     LLAMA: str = "llama"
#     MISTRAL: str = "mistral"
#     DEEPSEEK: str = "deepseek"


# class TRTLLMQuantizationType(Enum):
#     NO_QUANT: str = "no_quant"
#     WEIGHTS_ONLY_INT8: str = "weights_int8"
#     WEIGHTS_KV_INT8: str = "weights_kv_int8"
#     WEIGHTS_ONLY_INT4: str = "weights_int4"
#     WEIGHTS_KV_INT4: str = "weights_kv_int4"
#     SMOOTH_QUANT: str = "smooth_quant"
#     FP8: str = "fp8"
#     FP8_KV: str = "fp8_kv"

# class TrussTRTLLMPluginConfiguration(BaseModel):
#     multi_block_mode: bool = False
#     paged_kv_cache: bool = True
#     use_fused_mlp: bool = False

# class TrussTRTLLMBuildConfiguration(BaseModel):
#     base_model_architecture: TRTLLMModelArchitecture
#     max_input_len: int
#     max_output_len: int
#     max_batch_size: int
#     max_beam_width: int
#     max_prompt_embedding_table_size: int = 0
#     huggingface_ckpt_repository: Optional[str]
#     gather_all_token_logits: bool = False
#     strongly_typed: bool = False
#     quantization_type: TRTLLMQuantizationType = TRTLLMQuantizationType.NO_QUANT
#     tensor_parallel_count: int = 1
#     pipeline_parallel_count: int = 1
#     plugin_configuration: TrussTRTLLMPluginConfiguration = TrussTRTLLMPluginConfiguration()

# class TrussTRTLLMServingConfiguration(BaseModel):
#     engine_repository: str
#     tokenizer_repository: str
#     tensor_parallel_count: int = 1
#     pipeline_parallel_count: int = 1

# class TrussTRTLLMConfiguration(BaseModel):
#     serve: Optional[TrussTRTLLMServingConfiguration] = None
#     build: Optional[TrussTRTLLMBuildConfiguration] = None

#     @model_validator(mode="after")
#     def check_minimum_required_configuration(self):
#         if not self.serve and not self.build:
#             raise ValueError(
#                 "Either serve or build configurations must be provided"
#             )
#         if self.serve and self.build:
#             raise ValueError(
#                 "Both serve and build configurations cannot be provided"
#             )
#         if self.serve is not None:
#             if (self.serve.engine_repository is None) ^ (self.serve.tokenizer_repository is None):
#                 raise ValueError(
#                     "Both engine_repository and tokenizer_repository must be provided"
#                 )
#         return self

#     @property
#     def requires_build(self):
#         if self.build is not None:
#             return True
#         return False
