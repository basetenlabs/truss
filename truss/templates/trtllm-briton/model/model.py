import os
import signal
import socket
import subprocess
import threading
import time
from itertools import count

import briton_pb2
import briton_pb2_grpc
import grpc
from transformers import AutoTokenizer
from truss.config.trt_llm import TrussTRTLLMBuildConfiguration
from truss.constants import OPENAI_COMPATIBLE_TAG

BRITON_PORT = 50051

MODEL_INPUT_TO_BRITON_FIELD = {
    "max_tokens": "request_output_len",
    "beam_width": "beam_width",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "temperature": "temperature",
    "length_penalty": "len_penalty",
    "end_id": "end_id",
    "pad_id": "pad_id",
    "runtime_top_k": "runtime_top_k",
    "runtime_top_p": "runtime_top_p",
}


def is_port_available(port, host="localhost"):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Try to connect to the specified host and port
            s.bind((host, port))
            return True
    except OSError:
        # Port is not available
        return False


def briton_monitor(briton_process):
    while True:
        if briton_process.poll() is not None:
            print(
                f"Briton process has exited with code {briton_process.returncode}, exiting truss server"
            )
            pid = os.getpid()
            os.kill(pid, signal.SIGKILL)
        time.sleep(1)


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._config = kwargs["config"]
        self._data_dir = kwargs["data_dir"]
        self._stub = None
        self._secrets = kwargs["secrets"]
        self._request_id_counter = count(start=1)
        self._briton_process = None
        if OPENAI_COMPATIBLE_TAG not in self._config.get("model_metadata", {}):
            print(
                "Warning: Defaulting to openai compatible interface, even though not indicated in config."
            )

        if "trt_llm" not in self._config:
            raise ValueError("trt_llm config is required for this model")

        trtllm_config = self._config.get("trt_llm")
        truss_trtllm_build_config = TrussTRTLLMBuildConfiguration(
            **trtllm_config.get("build")
        )
        self._tp_count = truss_trtllm_build_config.tensor_parallel_count
        self._tokenizer_repository = (
            truss_trtllm_build_config.checkpoint_repository.repo
        )
        self._kv_cache_free_gpu_mem_fraction = (
            truss_trtllm_build_config.kv_cache_free_gpu_mem_fraction
        )

        self._hf_token = None
        try:
            self._hf_token = self._secrets.get("hf_access_token", None)
        except:  # noqa
            pass

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_repository, token=self._hf_token
        )
        # Start engine
        config_str = f"""
    engine_path: "{self._data_dir.resolve()}"
    hf_tokenizer: "{self._tokenizer_repository}"
    kv_cache_free_gpu_mem_fraction: {self._kv_cache_free_gpu_mem_fraction}
"""
        config_pbtxt_path = (self._data_dir / "briton_config.pbtxt").resolve()
        config_pbtxt_path.write_text(config_str)
        briton_env = os.environ.copy()
        if self._hf_token is not None:
            briton_env["HF_ACCESS_TOKEN"] = self._hf_token
        if self._tp_count is None or self._tp_count == 1:
            self._briton_process = subprocess.Popen(
                ["Briton", "--config", str(config_pbtxt_path)], env=briton_env
            )
        else:
            self._briton_process = subprocess.Popen(
                [
                    "mpirun",
                    "--allow-run-as-root",
                    "-n",
                    f"{self._tp_count}",
                    "Briton",
                    "--config",
                    str(config_pbtxt_path),
                ],
                env=briton_env,
            )
        while is_port_available(BRITON_PORT):
            print("Waiting for Briton to start")
            time.sleep(1)

        briton_monitor_thread = threading.Thread(
            target=briton_monitor, args=(self._briton_process,)
        )
        briton_monitor_thread.start()

    async def predict(self, model_input):
        """
        Run inference

        Note that the async nature of this function is a little tricky. Care is
        needed to make sure this function is a regular async function and not an
        async generator, i.e. there shouldn't be any direct yields in this
        function. This is because we need to support both streaming and
        non-streaming cases in this function. We do this by either returning an
        async-generator for the streaming case, or directly the full text for
        the other case. Returning an async generator for non-streaming case
        interferes with the open ai client proxy.
        """
        if self._stub is None:
            channel = grpc.aio.insecure_channel(f"localhost:{BRITON_PORT}")
            self._stub = briton_pb2_grpc.BritonStub(channel)

        prompt = model_input.get("prompt", None)
        if prompt is None and "messages" in model_input:
            messages = model_input.pop("messages")
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)

        request_id = int(str(os.getpid()) + str(next(self._request_id_counter)))
        request = briton_pb2.InferenceRequest(
            request_id=request_id,
            input_text=prompt,
        )
        # Set default end_id and pad_id
        if (
            hasattr(self._tokenizer, "eos_token_id")
            and self._tokenizer.eos_token_id is not None
        ):
            request.end_id = self._tokenizer.eos_token_id
        if (
            hasattr(self._tokenizer, "pad_token_id")
            and self._tokenizer.pad_token_id is not None
        ):
            request.pad_id = self._tokenizer.pad_token_id
        set_briton_request_fields_from_model_input(model_input, request)
        for words in ["bad_words", "stop_words"]:
            if words in model_input:
                for word in model_input[words].split(","):
                    getattr(request, words).append(word)

        resp_iter = self._stub.Infer(request)

        async def generate():
            async for response in resp_iter:
                yield response.output_text

        async def build_response():
            full_text = ""
            async for delta in resp_iter:
                full_text += delta.output_text
            return full_text

        try:
            if model_input.get("stream", True):
                return generate()
            else:
                return await build_response()
        except grpc.RpcError as ex:
            if ex.code() == grpc.StatusCode.INVALID_ARGUMENT:
                print(ex.details())
        except Exception as ex:
            print(f"An error has occurred: {ex}")
            raise ex


def set_briton_request_fields_from_model_input(model_input, briton_request):
    for model_input_key, briton_field in MODEL_INPUT_TO_BRITON_FIELD.items():
        if model_input_key in model_input:
            model_input_value = model_input[model_input_key]
            setattr(briton_request, briton_field, model_input_value)
