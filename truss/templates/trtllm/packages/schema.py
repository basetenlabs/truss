from typing import Any, Callable, Dict, Optional

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

    @staticmethod
    def from_bridge_oai_request(
        model_input: Dict[Any, Any],
        chat_templater: Callable[[Any], Any],
        request_id: str,
        eos_token_id: str,
    ):
        if "messages" not in model_input:
            raise ValueError("'messages' key expected in bridge request")

        messages = model_input.pop("messages")
        if "prompt" not in model_input:
            model_input["prompt"] = chat_templater(messages)

        # example of pulling off a value from raw
        # if 'raw' in model_input and isinstance(model_input['raw'], dict):
        #     seed = model_input['raw'].get(seed)
        return ModelInput(
            prompt=model_input.get("prompt"),
            max_tokens=model_input.get("max_tokens"),
            max_new_tokens=model_input.get("max_new_tokens"),
            temperature=model_input.get("temperature"),
            top_k=model_input.get("top_k"),
            top_p=model_input.get("top_p"),
            stop_words_list=model_input.get("stop_words_list"),
            repetition_penalty=model_input.get("repetition_penalty"),
            stream=model_input.get("stream"),
            eos_token_id=eos_token_id,
            request_id=request_id,
            # seed=seed,
        )

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
        output_len_data = np.ones_like(prompt_data, dtype=np.int32) * self._max_tokens
        bad_words_data = np.array([self._bad_words_list], dtype=object)
        stop_words_data = np.array([self._stop_words_list], dtype=object)
        # Note: this is necessary until we can serve trt-llm with this flag set to False.
        # as of this commit, the iterator will return nothing when set to false.
        stream_data = np.array([[True]], dtype=bool)
        beam_width_data = np.array([[self._beam_width]], dtype=np.int32)
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
            end_id_data = np.array([[self._eos_token_id]], dtype=np.int32)
            inputs.append(self._prepare_grpc_tensor("end_id", end_id_data))

        return inputs

"""
The BridgeCompletionRequest is input into the `from_bridge_request` function.
This model is json-serialized as input into the `predict` endpoint. 
If changes are needed, reach out the Baseten Core Product team

class BridgeCompletionRequest(BaseModel):
    messages: Union[str, List[Dict[str, str]]]
    # raw is the incoming request with the "messages" omitted
    # to avoid unnecessarily large request size
    raw: Dict[Any, Any]
    client_origin: str
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stop_words_list: Optional[List[str]] = None
"""