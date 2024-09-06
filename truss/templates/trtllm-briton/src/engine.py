import hashlib
import json
import os
import signal
import socket
import subprocess
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional

import briton_pb2
import briton_pb2_grpc
import grpc
from fastapi import HTTPException
from outlines.models.transformers import TransformerTokenizer
from outlines.processors.structured import JSONLogitsProcessor
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

# Use a directory that can be picked up by baseten-fs
FSM_CACHE_DIR = "/cache/model/fsm_cache"


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


class Engine:
    def __init__(self, **kwargs):
        self._loaded = False
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
        self._enable_kv_cache_reuse = (
            truss_trtllm_build_config.plugin_configuration.use_paged_context_fmha
        )

        self._hf_token = None
        try:
            self._hf_token = self._secrets.get("hf_access_token", None)
        except:  # noqa
            pass

        self._max_input_len = truss_trtllm_build_config.max_input_len
        self._max_beam_width = truss_trtllm_build_config.max_beam_width

    def load(self):
        if self._loaded:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_repository, token=self._hf_token
        )

        self._fsm_cache = FsmCache(Path(FSM_CACHE_DIR), self._tokenizer)

        # Start engine
        config_str = f"""
    engine_path: "{self._data_dir.resolve()}"
    hf_tokenizer: "{self._tokenizer_repository}"
    kv_cache_free_gpu_mem_fraction: {self._kv_cache_free_gpu_mem_fraction}
    enable_kv_cache_reuse: {"true" if self._enable_kv_cache_reuse else "false"}
    fsm_cache_dir: "{FSM_CACHE_DIR}"
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
        self._loaded = True

    def validate_input(self, model_input):
        beam_width = model_input.get("beam_width", None)
        # Beam width == 1.
        # There's no need to check if streaming is passed in the input.
        # Briton explicitly sets streaming to true in britonToTbRequest().
        # https://github.com/basetenlabs/baseten/blob/1c2c9cbe1adafc0c736566bd012abbe7d7e2c2da/briton/src/briton.cpp#L272
        if beam_width is not None and beam_width != 1:
            raise HTTPException(
                status_code=400, detail="TensorRT-LLM requires beam_width to equal 1"
            )

        # If Beam width != max_beam_width, TensorRt-LLM will fail an assert.
        # Since Briton sets streaming, the max_beam_width must aslo equal 1.
        if self._max_beam_width != 1:
            raise HTTPException(
                status_code=400,
                detail="TensorRT-LLM requires max_beam_width to equal 1.",
            )

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

        function_calling_schema = None
        tools = model_input.get("tools", None)
        if tools is not None:
            function_calling_schema = {
                "anyOf": [create_tool_schema(tool) for tool in tools],
            }

        prompt = model_input.get("prompt", None)
        if prompt is None and "messages" in model_input:
            messages = model_input.pop("messages")
            prompt = self._tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
        if prompt is None or len(prompt) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        self.validate_input(model_input)

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
        # Add output schema hash if we're function calling or response_format is provided
        schema_hash = None
        try:
            schema_hash = (
                self._fsm_cache.add_schema(function_calling_schema)
                if function_calling_schema is not None
                else self._fsm_cache.add_schema_from_input(model_input)
            )
        # If the input schema is invalid, we should return a 400
        except NotImplementedError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        if schema_hash is not None:
            request.output_schema_hash = schema_hash
        set_briton_request_fields_from_model_input(model_input, request)
        for words in ["bad_words", "stop_words"]:
            if words in model_input:
                for word in model_input[words].split(","):
                    getattr(request, words).append(word)

        resp_iter = self._stub.Infer(request)

        async def generate():
            eos_token = (
                self._tokenizer.eos_token
                if hasattr(self._tokenizer, "eos_token")
                else None
            )
            async for response in resp_iter:
                if eos_token:
                    yield response.output_text.removesuffix(eos_token)
                else:
                    yield response.output_text

        async def build_response():
            eos_token = (
                self._tokenizer.eos_token
                if hasattr(self._tokenizer, "eos_token")
                else None
            )
            full_text = ""
            async for delta in resp_iter:
                full_text += delta.output_text
            if eos_token:
                return full_text.removesuffix(eos_token)
            else:
                return full_text

        try:
            if model_input.get("stream", True):
                gen = generate()
                first_chunk = await gen.__anext__()

                async def generate_after_first_chunk():
                    yield first_chunk
                    async for chunk in gen:
                        yield chunk

                return generate_after_first_chunk()
            else:
                return await build_response()
        except grpc.RpcError as ex:
            if ex.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise HTTPException(status_code=400, detail=ex.details())
            # If the error is another GRPC exception like NotImplemented, we should return a 500
            else:
                raise HTTPException(
                    status_code=500, detail=f"An error has occurred: {ex}"
                )
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"An error has occurred: {ex}")


def create_tool_schema(tool_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"const": tool_json["function"]["name"]},
            "parameters": tool_json["function"]["parameters"],
        },
        "required": ["name", "parameters"],
    }


class FsmCache:
    def __init__(self, cache_dir: Path, tokenizer: AutoTokenizer):
        self._cache_dir = cache_dir
        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = set(f.name for f in self._cache_dir.iterdir() if f.is_file())
        self._tokenizer = tokenizer

    def add_schema(self, schema: Dict[str, Any]) -> str:
        schema_str = json.dumps(schema)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
        if schema_hash not in self._cache:
            fsm = self._create_fsm(schema)
            (self._cache_dir / schema_hash).write_bytes(fsm.SerializeToString())
            self._cache.add(schema_hash)
        return schema_hash

    def add_schema_from_input(self, model_input: Dict[str, Any]) -> Optional[str]:
        schema_hash = None
        schema = self._extract_schema(model_input)
        if schema is not None:
            schema_hash = self.add_schema(schema)
        return schema_hash

    def _create_fsm(self, schema: Dict[str, Any]) -> briton_pb2.StatesToTokens:  # type: ignore[name-defined]
        outlines_tokenizer = TransformerTokenizer(self._tokenizer)
        logits_processor = JSONLogitsProcessor(schema, outlines_tokenizer)
        guide = logits_processor.fsm

        states_to_tokens = {}
        for state, token_to_next_state in guide.states_to_token_maps.items():
            states_to_tokens[state] = briton_pb2.TokenToNextState(  # type: ignore[attr-defined]
                token_to_next_state=token_to_next_state
            )
        states_to_tokens_pb = briton_pb2.StatesToTokens(  # type: ignore[attr-defined]
            states_to_tokens=states_to_tokens,
            vocab_size=len(self._tokenizer.vocab),
            eos_token_id=self._tokenizer.eos_token_id,
        )
        return states_to_tokens_pb

    @staticmethod
    def _extract_schema(model_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "response_format" not in model_input:
            return None
        response_format = model_input["response_format"]
        if "type" not in response_format or response_format["type"] != "json_schema":
            raise HTTPException(
                status_code=400,
                detail='response_format["type"] must be json_schema.',
            )
        if "json_schema" not in response_format:
            raise HTTPException(
                status_code=400,
                detail='response_format["json_schema"] must be provided.',
            )
        json_schema = response_format["json_schema"]
        if "schema" not in json_schema:
            raise HTTPException(
                status_code=400,
                detail='response_format["json_schema"]["schema"] must be provided.',
            )
        return json_schema["schema"]


def set_briton_request_fields_from_model_input(model_input, briton_request):
    for model_input_key, briton_field in MODEL_INPUT_TO_BRITON_FIELD.items():
        if model_input_key in model_input:
            model_input_value = model_input[model_input_key]
            setattr(briton_request, briton_field, model_input_value)
