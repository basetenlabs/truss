import os
from itertools import count

from schema import ModelInput
from transformers import AutoTokenizer
from triton_client import TritonClient, TritonServer
from truss.config.trt_llm import TrussTRTLLMBuildConfiguration
from truss.constants import OPENAI_COMPATIBLE_TAG

from constants import (
    GRPC_SERVICE_PORT,
    HF_AUTH_KEY_CONSTANT,
    HTTP_SERVICE_PORT,
    TOKENIZER_KEY_CONSTANT,
)

DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_NEW_TOKENS = 500


class Model:
    def __init__(self, data_dir, config, secrets, lazy_data_resolver):
        self._data_dir = data_dir
        self._config = config
        self._secrets = secrets
        self._request_id_counter = count(start=1)
        self._lazy_data_resolver = lazy_data_resolver
        self.triton_client = None
        self.triton_server = None
        self.tokenizer = None
        self.uses_openai_api = None

    def load(self):
        trtllm_config = self._config.get("trt_llm", {})
        self.uses_openai_api = OPENAI_COMPATIBLE_TAG in self._config.get(
            "model_metadata", {}
        ).get("tags", [])
        hf_access_token = None
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]

        engine_repository_path = self._data_dir
        truss_trtllm_build_config = TrussTRTLLMBuildConfiguration(
            **trtllm_config.get("build")
        )
        tokenizer_repository = truss_trtllm_build_config.checkpoint_repository.repo
        tensor_parallel_count = truss_trtllm_build_config.tensor_parallel_count
        pipeline_parallel_count = truss_trtllm_build_config.pipeline_parallel_count
        world_size = tensor_parallel_count * pipeline_parallel_count

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repository, token=hf_access_token
        )
        self.eos_token_id = self.tokenizer.eos_token_id

        # Set up Triton Server
        env = {}
        if hf_access_token:
            env[HF_AUTH_KEY_CONSTANT] = hf_access_token
        env[TOKENIZER_KEY_CONSTANT] = tokenizer_repository

        self.triton_server = TritonServer(
            grpc_port=GRPC_SERVICE_PORT,
            http_port=HTTP_SERVICE_PORT,
        )
        self.triton_server.create_model_repository(
            truss_data_dir=self._data_dir,
            engine_repository_path=engine_repository_path,
            huggingface_auth_token=hf_access_token,
        )
        self.triton_server.start(
            world_size=world_size,
            env=env,
        )
        self.triton_client = TritonClient(
            grpc_service_port=GRPC_SERVICE_PORT,
        )

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

        async def build_response():
            full_text = ""
            async for delta in result_iterator:
                full_text += delta
            return full_text

        if model_input.stream:
            return generate()
        else:
            text = await build_response()
            if self.uses_openai_api:
                return text
            else:
                return {"text": text}
