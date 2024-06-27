import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import tritonclient.grpc.aio as grpcclient
import tritonclient.http as httpclient
from schema import ModelInput
from utils import download_engine, prepare_model_repository

from constants import (
    ENTRYPOINT_MODEL_NAME,
    GRPC_SERVICE_PORT,
    TENSORRT_LLM_MODEL_REPOSITORY_PATH,
)


class TritonServer:
    def __init__(self, grpc_port: int = 8001, http_port: int = 8003):
        self.grpc_port = grpc_port
        self.http_port = http_port
        self._server_process = None

    def create_model_repository(
        self,
        truss_data_dir: Path,
        engine_repository_path: Optional[str] = None,
        huggingface_auth_token: Optional[str] = None,
    ) -> None:
        if engine_repository_path:
            if Path(engine_repository_path).is_dir():
                if str(engine_repository_path) != str(truss_data_dir):
                    shutil.copytree(
                        engine_repository_path, truss_data_dir, dirs_exist_ok=True
                    )
            else:
                # If engine_repository_path is not a local directory, download the engine
                download_engine(
                    engine_repository=engine_repository_path,
                    fp=truss_data_dir,
                    auth_token=huggingface_auth_token,
                )

        prepare_model_repository(truss_data_dir)
        return

    def start(self, world_size: int = 1, env: dict = {}) -> None:
        mpirun_command = ["mpirun", "--allow-run-as-root"]
        mpi_commands = []
        for i in range(world_size):
            mpi_command = [
                "-n",
                "1",
                "tritonserver",
                f"--model-repository={TENSORRT_LLM_MODEL_REPOSITORY_PATH}",
                f"--grpc-port={str(self.grpc_port)}",
                f"--http-port={str(self.http_port)}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
                ":",
            ]

            mpi_commands.extend(mpi_command)
        command = mpirun_command + mpi_commands

        self._server_process = subprocess.Popen(  # type: ignore
            command,
            env={**os.environ, **env},
        )
        while not self.is_alive and not self.is_ready:
            time.sleep(2)
        return

    def stop(self):
        if self._server_process:
            if self.is_ready:
                self._server_process.kill()
            self._server_process = None
        return

    @property
    def is_alive(self) -> bool:
        try:
            http_client = httpclient.InferenceServerClient(
                url=f"localhost:{self.http_port}", verbose=False
            )
            return http_client.is_server_live()
        except ConnectionRefusedError:
            return False

    @property
    def is_ready(self) -> bool:
        try:
            http_client = httpclient.InferenceServerClient(
                url=f"localhost:{self.http_port}", verbose=False
            )
            return http_client.is_model_ready(model_name=ENTRYPOINT_MODEL_NAME)
        except ConnectionRefusedError:
            return False


class TritonClient:
    def __init__(self, grpc_service_port: int = GRPC_SERVICE_PORT):
        self.grpc_service_port = grpc_service_port
        self._grpc_client = None

    def start_grpc_stream(self) -> grpcclient.InferenceServerClient:
        if self._grpc_client:
            return self._grpc_client

        self._grpc_client = grpcclient.InferenceServerClient(
            url=f"localhost:{self.grpc_service_port}", verbose=False
        )
        return self._grpc_client

    async def infer(
        self, model_input: ModelInput, model_name=ENTRYPOINT_MODEL_NAME
    ) -> AsyncGenerator[str, None]:
        grpc_client_instance = self.start_grpc_stream()
        inputs = model_input.to_tensors()

        async def input_generator():
            yield {
                "model_name": model_name,
                "inputs": inputs,
                "request_id": model_input.request_id,
            }

        response_iterator = grpc_client_instance.stream_infer(
            inputs_iterator=input_generator(),
        )

        try:
            async for response in response_iterator:
                result, error = response
                if result:
                    result = result.as_numpy("text_output")
                    yield result[0].decode("utf-8")
                else:
                    yield json.dumps({"status": "error", "message": error.message()})

        except grpcclient.InferenceServerException as e:
            print(f"InferenceServerException: {e}")
