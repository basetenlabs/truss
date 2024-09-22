import asyncio
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional

import briton_pb2
import briton_pb2_grpc
import grpc
from fastapi import HTTPException
from outlines.models.transformers import TransformerTokenizer
from outlines.processors.structured import JSONLogitsProcessor
from transformers import AutoTokenizer, PreTrainedTokenizerFast
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

TOOL_CALL_IDS = {
    "llama": 128010,
    "mistral": 5,
}

TOOL_CALL_TOKENS = {
    "llama": "<|python_tag|>",
    "mistral": "[TOOL_CALLS]",
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
        self._base_model = truss_trtllm_build_config.base_model
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

        # TODO(@bdubayah): configure this based on CPU. But os.cpu_count() returns the
        # number of CPUs for the entire node, not just the container.
        self._max_fsm_workers = 10
        print(f"Using {self._max_fsm_workers} workers for FSM schema generation")

    def load(self):
        if self._loaded:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_repository, token=self._hf_token
        )

        self._fsm_cache = FsmCache(
            Path(FSM_CACHE_DIR), self._tokenizer, self._max_fsm_workers
        )

        # We only support Llama and mistral with Briton, for which this should
        # apply.
        assert isinstance(self._tokenizer, PreTrainedTokenizerFast)

        # These are tokens outside of tokenizer.json. We need to pass these to
        # Briton, to pass to rust tokenizer.
        added_token_decoders = self._tokenizer.added_tokens_decoder
        added_tokens = [token for token in added_token_decoders.values()]

        self._saved_tokenizer_dir = str(self._data_dir / "saved_tokenizer")
        self._tokenizer.save_pretrained(self._saved_tokenizer_dir)

        # Pass tokenizer file to Briton for the rust tokenizer.
        tokenizer_file = Path(self._saved_tokenizer_dir) / "tokenizer.json"
        config_str = f"""
engine_path: "{self._data_dir.resolve()}"
hf_tokenizer: "{tokenizer_file.resolve()}"
kv_cache_free_gpu_mem_fraction: {self._kv_cache_free_gpu_mem_fraction}
enable_kv_cache_reuse: {"true" if self._enable_kv_cache_reuse else "false"}
fsm_cache_dir: "{FSM_CACHE_DIR}"
"""

        # Pass added tokens to Briton for the rust tokenizer.
        if len(added_tokens) > 0:
            config_str += "\n" + "\n".join(
                _serialize_added_tokens_to_config(added_tokens)
            )

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

        # TODO(@bdubayah): refactor into smaller functions
        function_calling_schema = None
        tools = model_input.get("tools")
        tool_choice = model_input.get("tool_choice")
        force_tools = None
        if tool_choice is not None:
            if not (
                tool_choice in ["none", "required", "auto"]
                or isinstance(tool_choice, dict)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="tool_choice must be 'none', 'required', 'auto', or an object of the form {'type': 'function', 'function': {'name': 'function_name'}}.",
                )
            if tool_choice == "none":
                tools = None
                tool_choice = None
            elif tool_choice == "required":
                if tools is None:
                    raise HTTPException(
                        status_code=400,
                        detail="tool_choice is 'required' but no tools provided.",
                    )
                force_tools = True
        if tools is not None:
            if model_input.get("response_format") is not None:
                raise HTTPException(
                    status_code=400,
                    detail="response_format is not allowed when tools are provided, unless tool_choice is 'none'.",
                )
            tool_schemas = {
                tool["function"]["name"]: create_tool_schema(tool) for tool in tools
            }
            if isinstance(tool_choice, dict):
                if tool_choice.get("type") != "function":
                    raise HTTPException(
                        status_code=400, detail="tool_choice['type'] must be function."
                    )
                if tool_choice.get("function") is None:
                    raise HTTPException(
                        status_code=400, detail="tool_choice['function'] required."
                    )
                if not isinstance(tool_choice["function"], dict):
                    raise HTTPException(
                        status_code=400,
                        detail="tool_choice['function'] must be an object.",
                    )
                if tool_choice["function"].get("name") is None:
                    raise HTTPException(
                        status_code=400,
                        detail="tool_choice['function']['name'] required.",
                    )
                if tool_choice["function"]["name"] not in tool_schemas:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Tool choice function {tool_choice['function']['name']} not in tools.",
                    )
                tool_schemas = {
                    tool_choice["function"]["name"]: tool_schemas[
                        tool_choice["function"]["name"]
                    ]
                }
                force_tools = True
            elif tool_choice is None or tool_choice == "auto":
                force_tools = False
            function_calling_schema = {
                "type": "array",
                "items": {
                    "anyOf": list(tool_schemas.values()),
                },
                "minItems": 1,
            }

        prompt = model_input.get("prompt", None)
        if prompt is not None and tools is not None:
            raise HTTPException(
                status_code=400,
                detail="tools can only be provided in chat mode. Please set messages instead of prompt, remove tools, or set tool_choice to 'none'.",
            )
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
                await self._fsm_cache.add_schema(function_calling_schema)
                if function_calling_schema is not None
                else await self._fsm_cache.add_schema_from_input(model_input)
            )
        # If the input schema is invalid, we should return a 400
        except NotImplementedError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        if schema_hash is not None:
            request.output_schema_hash = schema_hash
        if force_tools is not None:
            request.tools_id = TOOL_CALL_IDS[self._base_model]
            request.force_tools = force_tools
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
            tool_call_token = TOOL_CALL_TOKENS.get(self._base_model)
            async for response in resp_iter:
                output_text = response.output_text
                if tool_call_token:
                    output_text = output_text.removeprefix(tool_call_token)
                if eos_token:
                    output_text = output_text.removesuffix(eos_token)
                yield output_text

        async def build_response():
            eos_token = (
                self._tokenizer.eos_token
                if hasattr(self._tokenizer, "eos_token")
                else None
            )
            tool_call_token = TOOL_CALL_TOKENS.get(self._base_model)
            full_text = ""
            async for delta in resp_iter:
                full_text += delta.output_text
            if tool_call_token:
                full_text = full_text.removeprefix(tool_call_token)
            if eos_token:
                full_text = full_text.removesuffix(eos_token)
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
            if (
                ex.code() == grpc.StatusCode.INVALID_ARGUMENT
                or ex.code() == grpc.StatusCode.UNIMPLEMENTED
            ):
                raise HTTPException(status_code=400, detail=ex.details())
            # If the error is another type of gRPC error, we should return a 500
            else:
                raise HTTPException(
                    status_code=500, detail=f"An error has occurred: {ex.details()}"
                )
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"An error has occurred: {ex}")


def _serialize_added_tokens_to_config(added_tokens: list) -> List[str]:
    """Serialize to pbtxt format."""
    lines = ["added_tokens {"]
    for added_token in added_tokens:
        token_lines = _serialize_added_token_to_config(added_token)
        lines.extend([f"  {line}" for line in token_lines])
    lines.append("}")
    return lines


def _serialize_added_token_to_config(added_token) -> List[str]:
    """Serialize to pbtxt format."""
    fields = [
        f'content: "{added_token.content}"',
        f"single_word: {added_token.single_word}",
        f"lstrip: {added_token.lstrip}",
        f"rstrip: {added_token.rstrip}",
        f"normalized: {added_token.normalized}",
        f"special: {added_token.special}",
    ]
    return [
        "tokens {",
        *[f"  {field}" for field in fields],
        "}",
    ]


def create_tool_schema(tool_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"const": tool_json["function"]["name"]},
            "parameters": tool_json["function"]["parameters"],
        },
        "required": ["name", "parameters"],
    }


outlines_tokenizer = None


def worker(vocab_size: int, end_id: int, schema: Dict[str, Any], output_path: Path):
    logits_processor = JSONLogitsProcessor(schema, outlines_tokenizer)
    guide = logits_processor.fsm
    states_to_tokens = {}
    for state, token_to_next_state in guide.states_to_token_maps.items():
        states_to_tokens[state] = briton_pb2.TokenToNextState(  # type: ignore[attr-defined]
            token_to_next_state=token_to_next_state
        )
    states_to_tokens_pb = briton_pb2.StatesToTokens(  # type: ignore[attr-defined]
        states_to_tokens=states_to_tokens,
        vocab_size=vocab_size,
        eos_token_id=end_id,
    )
    if not output_path.exists():
        try:
            # Open the file with flags to protect against concurrent writes.
            # O_CREAT: Create the file if it does not exist.
            # O_EXCL: Ensure that this call creates the file exclusively. If the file already exists, the call will fail.
            # O_WRONLY: Open the file for write-only access.
            fd = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(states_to_tokens_pb.SerializeToString())
                fcntl.flock(f, fcntl.LOCK_UN)
        except FileExistsError:
            pass


def dummy_task():
    pass


class FsmCache:
    def __init__(self, cache_dir: Path, tokenizer: AutoTokenizer, max_workers: int):
        self._cache_dir = cache_dir
        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = set(f.name for f in self._cache_dir.iterdir() if f.is_file())
        self._lock = threading.Lock()
        self._tokenizer = tokenizer

        # Concurrent FSM generation initialization
        # Make sure we fork because (1) it's faster and (2) it seems that spawning
        # ends up being sequential
        multiprocessing.set_start_method("fork", force=True)
        global outlines_tokenizer
        outlines_tokenizer = TransformerTokenizer(tokenizer)
        # This is very important. The first time JSONLogitsProcessor is called, some library-wide
        # initializations are done in memory (that take 5s). By doing it before we fork, we avoid paying
        # that cost for each forked process.
        _ = JSONLogitsProcessor({"properties": {}}, outlines_tokenizer)
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        # We must create all processes BEFORE the GRPC python client is started to avoid errors
        # forking from the process GRPC is running in
        for _ in range(max_workers):
            self._executor.submit(dummy_task)

    async def add_schema(self, schema: Dict[str, Any]) -> str:
        schema_str = json.dumps(schema)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
        if schema_hash not in self._cache:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                worker,
                len(self._tokenizer.vocab),
                self._tokenizer.eos_token_id,
                schema,
                self._cache_dir / schema_hash,
            )
            with self._lock:
                self._cache.add(schema_hash)
        return schema_hash

    async def add_schema_from_input(self, model_input: Dict[str, Any]) -> Optional[str]:
        schema_hash = None
        schema = self._extract_schema(model_input)
        if schema is not None:
            schema_hash = await self.add_schema(schema)
        return schema_hash

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
