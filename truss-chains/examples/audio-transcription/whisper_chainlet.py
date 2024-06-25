# This is the Chainlet corresponding to the Truss model:
# https://github.com/basetenlabs/truss-examples/tree/main/07-high-performance-dynamic-batching

import base64
import tempfile
from typing import re

import data_types
import truss_chains as chains
from huggingface_hub import snapshot_download
from truss import truss_config

# Max queue time is the amount of time in seconds to wait to fill the batch
MAX_QUEUE_TIME = 0.25

# Maximum size of the batch. This is dictated by the compiled engine.
MAX_BATCH_SIZE = 8


def base64_to_wav(base64_string, output_file_path):
    binary_data = base64.b64decode(base64_string)
    with open(output_file_path, "wb") as wav_file:
        wav_file.write(binary_data)
    return output_file_path


@chains.mark_entrypoint
class WhisperModel(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            base_image="baseten/trtllm-server:r23.12_baseten_v0.9.0.dev2024022000",
            apt_requirements=[
                "python3.10-venv",
                "openmpi-bin",
                "libopenmpi-dev",
                "ffmpeg",
            ],
            pip_requirements=[
                "git+https://github.com/basetenlabs/truss.git@marius/chains",
                "--extra-index-url https://pypi.nvidia.com",
                "async-batcher>=0.2.0",
                "hf_transfer",
                "huggingface_hub==0.20.3",
                "kaldialign==0.9",
                "openai-whisper==20231117",
                "pydantic>=2.7.1",
                "safetensors",
                "soundfile==0.12.1",
                "tensorrt_llm==0.10.0.dev2024042300",
                "tiktoken==0.6.0",
                "torchaudio",
            ],
        ),
        compute=chains.Compute(gpu="A10G", predict_concurrency=128),
        assets=chains.Assets(
            # secret_keys=["hf_access_token"]
            external_data=[
                truss_config.ExternalDataItem(
                    local_data_path="assets/multilingual.tiktoken",
                    url="https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
                ),
                truss_config.ExternalDataItem(
                    local_data_path="assets/mel_filters.npz",
                    url="https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
                ),
            ]
        ),
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._data_dir = context.data_dir
        snapshot_download(
            repo_id="baseten/trtllm-whisper-a10g-large-v2-1",
            local_dir=context.data_dir,
            # allow_patterns=["**"],
            # token=context.secrets["hf_access_token"],
        )
        from whisper_lib import run

        self._model = run.WhisperTRTLLM(f"{context.data_dir}")
        self._batcher = run.MlBatcher(
            model=self._model,
            max_batch_size=MAX_BATCH_SIZE,
            max_queue_time=MAX_QUEUE_TIME,
        )

    async def run_remote(
        self, whisper_input: data_types.WhisperInput
    ) -> data_types.WhisperResult:
        import torch
        from whisper_lib import whisper_utils

        with tempfile.NamedTemporaryFile() as fp:
            base64_to_wav(whisper_input.audio_b64, fp.name)
            mel, total_duration = whisper_utils.log_mel_spectrogram(
                fp.name,
                self._model.n_mels,
                device="cuda",
                return_duration=True,
                mel_filters_dir=f"{self._data_dir}/assets",
            )
            mel = mel.type(torch.float16)
            mel = mel.unsqueeze(0)
            prediction = await self._batcher.process(item=mel)

            # remove all special tokens in the prediction
            prediction = re.sub(r"<\|.*?\|>", "", prediction)
            return data_types.WhisperResult(
                segments=[
                    data_types.WhisperSegment(
                        start_time_sec=0,
                        end_time_sec=total_duration,
                        text=prediction.strip(),
                    )
                ],
            )
