import os
from itertools import count

import build_engine_utils
from constants import (
    GRPC_SERVICE_PORT,
    HF_AUTH_KEY_CONSTANT,
    HTTP_SERVICE_PORT,
    TOKENIZER_KEY_CONSTANT,
)
from schema import ModelInput, TrussBuildConfig
from transformers import AutoTokenizer
from triton_client import TritonClient, TritonServer


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
        build_config = TrussBuildConfig(**self._config["build"]["arguments"])
        self.uses_openai_api = "openai-compatible" in self._config.get(
            "model_metadata", {}
        ).get("tags", [])
        hf_access_token = None
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]

        # TODO(Abu): Move to pre-runtime
        if build_config.requires_build:
            build_engine_utils.build_engine_from_config_args(
                engine_build_args=build_config.engine_build_args,
                dst=self._data_dir,
            )

        self.triton_server = TritonServer(
            grpc_port=GRPC_SERVICE_PORT,
            http_port=HTTP_SERVICE_PORT,
        )

        self.triton_server.create_model_repository(
            truss_data_dir=self._data_dir,
            engine_repository_path=build_config.engine_repository
            if not build_config.requires_build
            else None,
            huggingface_auth_token=hf_access_token,
        )

        env = {}
        if hf_access_token:
            env[HF_AUTH_KEY_CONSTANT] = hf_access_token
        env[TOKENIZER_KEY_CONSTANT] = build_config.tokenizer_repository

        self.triton_server.start(
            tensor_parallelism=build_config.tensor_parallel_count,
            env=env,
        )

        self.triton_client = TritonClient(
            grpc_service_port=GRPC_SERVICE_PORT,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            build_config.tokenizer_repository, token=hf_access_token
        )
        self.eos_token_id = self.tokenizer.eos_token_id

    async def predict(self, model_input):
        model_input["request_id"] = str(os.getpid()) + str(
            next(self._request_id_counter)
        )
        model_input["eos_token_id"] = self.eos_token_id

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
