import json
import os
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import tritonclient.grpc.aio as grpcclient
import tritonclient.http as httpclient
from utils import (
    GRPC_SERVICE_PORT,
    HTTP_SERVICE_PORT,
    prepare_grpc_tensor,
    prepare_model_repository,
    server_loaded,
)


class TritonClient:
    def __init__(self, data_dir: Path, model_repository_dir: Path, parallel_count=1):
        self._data_dir: Path = data_dir
        self._model_repository_dir: Path = model_repository_dir
        self._parallel_count = parallel_count
        self._http_client: Optional[httpclient.InferenceServerClient] = None
        self.grpc_client_instance: Optional[grpcclient.InferenceServerClient] = None

    def start_grpc_stream(self) -> grpcclient.InferenceServerClient:
        if self.grpc_client_instance:
            return self.grpc_client_instance

        return grpcclient.InferenceServerClient(
            url=f"localhost:{GRPC_SERVICE_PORT}", verbose=False
        )

    def load_server_and_model(self, env: dict) -> None:
        """Loads the Triton server and the model."""
        if not server_loaded():
            prepare_model_repository(self._data_dir)
            self.start_server(mpi=self._parallel_count, env=env)

        self._http_client = httpclient.InferenceServerClient(
            url=f"localhost:{HTTP_SERVICE_PORT}", verbose=False
        )

        is_server_up = False
        while not is_server_up:
            try:
                is_server_up = self._http_client.is_server_live()
            except ConnectionRefusedError:
                time.sleep(2)
                continue

        while not self._http_client.is_model_ready(model_name="ensemble"):
            time.sleep(2)
            continue

    def build_server_start_command(self, mpi: int = 1, env: dict = {}) -> list:
        base_command = [
            "tritonserver",
            "--model-repository",
            str(self._model_repository_dir),
            "--grpc-port",
            str(GRPC_SERVICE_PORT),
            "--http-port",
            str(HTTP_SERVICE_PORT),
        ]

        if mpi == 1:
            return base_command

        mpirun_command = ["mpirun", "--allow-run-as-root"]
        # Generate mpi_commands with a unique shm-region-prefix-name for each MPI process
        mpi_commands = []
        for i in range(mpi):
            mpi_command = [
                "-n",
                "1",
                *base_command,
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{str(i)}_",
            ]
            mpi_commands.append(" ".join(mpi_command))

        # Join the individual mpi commands with ' : ' as required by mpirun syntax for multiple commands
        combined_mpi_commands = " : ".join(mpi_commands)

        return mpirun_command + [combined_mpi_commands]

    def start_server(
        self,
        mpi: int = 1,
        env: dict = {},
    ) -> subprocess.Popen:
        """Triton Inference Server has different startup commands depending on
        whether it is running in a TP=1 or TP>1 configuration. This function
        starts the server with the appropriate command."""

        command = self.build_server_start_command(mpi, env)
        return subprocess.Popen(
            command,
            env={**os.environ, **env},
        )

    async def infer(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 50,
        beam_width: int = 1,
        bad_words: list = [""],
        stop_words: list = [""],
        stream: bool = True,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        eos_token_id: Optional[int] = None,
        model_name="ensemble",
    ) -> AsyncGenerator[str, None]:
        """Infer a response from the model."""
        prompt_data = np.array([[prompt]], dtype=object)
        output_len_data = np.ones_like(prompt_data).astype(np.uint32) * max_tokens
        bad_words_data = np.array([bad_words], dtype=object)
        stop_words_data = np.array([stop_words], dtype=object)
        stream_data = np.array([[stream]], dtype=bool)
        beam_width_data = np.array([[beam_width]], dtype=np.uint32)
        repetition_penalty_data = np.array([[repetition_penalty]], dtype=np.float32)

        inputs = [
            prepare_grpc_tensor("text_input", prompt_data),
            prepare_grpc_tensor("max_tokens", output_len_data),
            prepare_grpc_tensor("bad_words", bad_words_data),
            prepare_grpc_tensor("stop_words", stop_words_data),
            prepare_grpc_tensor("stream", stream_data),
            prepare_grpc_tensor("beam_width", beam_width_data),
            prepare_grpc_tensor("repetition_penalty", repetition_penalty_data),
        ]

        if not ignore_eos:
            assert (
                eos_token_id is not None
            ), "eos_token_id must be provided if ignore_eos is False"
            end_id_data = np.array([[eos_token_id]], dtype=np.uint32)
            inputs.append(prepare_grpc_tensor("end_id", end_id_data))

        async def input_generator():
            yield {"model_name": model_name, "inputs": inputs, "request_id": request_id}

        self.grpc_client_instance = self.start_grpc_stream()
        result_iterator = self.grpc_client_instance.stream_infer(
            inputs_iterator=input_generator(),
        )

        async for response in result_iterator:
            result, error = response
            if result:
                result = result.as_numpy("text_output")
                yield result[0].decode("utf-8")
            else:
                yield json.dumps({"status": "error", "message": error.message()})
