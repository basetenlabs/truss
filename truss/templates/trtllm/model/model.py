import os
from itertools import count

import build_engine_utils
from builder.types import TrussTRTLLMConfiguration
from constants import (
    GRPC_SERVICE_PORT,
    HF_AUTH_KEY_CONSTANT,
    HTTP_SERVICE_PORT,
    TOKENIZER_KEY_CONSTANT,
)
from schema import ModelInput
from transformers import AutoTokenizer
from triton_client import TritonClient, TritonServer
from utils import execute_command

DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_NEW_TOKENS = 500


class Model:
    def __init__(self, data_dir, config, secrets):
        self._data_dir = data_dir
        self._config = config
        self._secrets = secrets
        self._request_id_counter = count(start=1)
        self.triton_client = None
        self.triton_server = None
        self.tokenizer = None
        self.uses_openai_api = None

    def load(self):
        execute_command(["ldconfig"])
        trtllm_config = TrussTRTLLMConfiguration(**self._config.get("trt_llm", {}))
        self.uses_openai_api = "openai-compatible" in self._config.get(
            "model_metadata", {}
        ).get("tags", [])
        hf_access_token = None
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]

        # The underlying engine build call is idempotent, so we can call it in the load
        # even if the engine is already built. The engine build call checks to see if
        # there are .engine files in the destination directory and skips the build if
        # they are present.
        if trtllm_config.requires_build:
            build_engine_utils.build_engine_from_config_args(
                truss_trtllm_configuration=trtllm_config,
                checkpoint_dir_path=None,
                dst=self._data_dir,
            )

        self.triton_server = TritonServer(
            grpc_port=GRPC_SERVICE_PORT,
            http_port=HTTP_SERVICE_PORT,
        )

        if not trtllm_config.requires_build:
            engine_repository_path = trtllm_config.serve.engine_repository
            tokenizer_repository = trtllm_config.serve.tokenizer_repository
            tensor_parallel_count = trtllm_config.serve.tensor_parallel_count
            pipeline_parallel_count = trtllm_config.serve.pipeline_parallel_count
        else:
            # If this model required a build, the engine live inside the data_dir
            engine_repository_path = self._data_dir
            tokenizer_repository = trtllm_config.build.huggingface_ckpt_repository
            tensor_parallel_count = trtllm_config.build.tensor_parallel_count
            pipeline_parallel_count = trtllm_config.build.pipeline_parallel_count

        world_size = tensor_parallel_count * pipeline_parallel_count

        self.triton_server.create_model_repository(
            truss_data_dir=self._data_dir,
            engine_repository_path=engine_repository_path,
            huggingface_auth_token=hf_access_token,
        )

        env = {}
        if hf_access_token:
            env[HF_AUTH_KEY_CONSTANT] = hf_access_token
        env[TOKENIZER_KEY_CONSTANT] = tokenizer_repository

        self.triton_server.start(
            world_size=world_size,
            env=env,
        )

        self.triton_client = TritonClient(
            grpc_service_port=GRPC_SERVICE_PORT,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repository, token=hf_access_token
        )
        self.eos_token_id = self.tokenizer.eos_token_id

    async def predict(self, model_input):
        if "messages" not in model_input and "prompt" not in model_input:
            raise ValueError("Prompt or messages must be provided")

        model_input.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        model_input.setdefault("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        model_input["request_id"] = str(os.getpid()) + str(
            next(self._request_id_counter)
        )
        model_input["eos_token_id"] = self.eos_token_id

        if "messages" in model_input:
            messages = model_input.pop("messages")
            if self.uses_openai_api and "prompt" not in model_input:
                model_input["prompt"] = self.tokenizer.apply_chat_template(
                    messages, tokenize=False
                )

        self.triton_client.start_grpc_stream()
        model_input = ModelInput(**model_input)
        result_iterator = self.triton_client.infer(model_input)

        async def generate():
            async for result in result_iterator:
                yield result

        if model_input.stream:
            return generate()
        else:
            if self.uses_openai_api:
                return "".join(generate())
            else:
                return {"text": "".join(generate())}
