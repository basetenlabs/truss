# flake8: noqa F402
# This location assumes `fde`-repo is checked out at the same level as `truss`-repo.
_LOCAL_WHISPER_LIB = "../../../../fde/whisper-trt/src"
import sys

sys.path.append(_LOCAL_WHISPER_LIB)

import base64

import data_types
import truss_chains as chains
from huggingface_hub import snapshot_download


@chains.mark_entrypoint
class WhisperModel(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            base_image="baseten/truss-server-base:3.10-gpu-v0.9.0",
            apt_requirements=["python3.10-venv", "openmpi-bin", "libopenmpi-dev"],
            pip_requirements=[
                "--extra-index-url https://pypi.nvidia.com",
                "tensorrt_llm==0.10.0.dev2024042300",
                "hf_transfer",
                "janus",
                "kaldialign",
                "librosa",
                "mpi4py==3.1.4",
                "safetensors",
                "soundfile",
                "tiktoken",
                "torchaudio",
                "async-batcher>=0.2.0",
                "pydantic>=2.7.1",
            ],
            external_package_dirs=[chains.make_abs_path_here(_LOCAL_WHISPER_LIB)],
        ),
        compute=chains.Compute(gpu="A10G", predict_concurrency=128),
        assets=chains.Assets(secret_keys=["hf_access_token"]),
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        snapshot_download(
            repo_id="baseten/whisper_trt_large-v3_A10G_i224_o512_bs8_bw5",
            local_dir=context.data_dir,
            allow_patterns=["**"],
            token=context.secrets["hf_access_token"],
        )
        from whisper_trt import WhisperModel

        self._model = WhisperModel(str(context.data_dir), max_queue_time=0.050)

    async def run_remote(
        self, whisper_input: data_types.WhisperInput
    ) -> data_types.WhisperResult:
        binary_data = base64.b64decode(whisper_input.audio_b64.encode("utf-8"))
        waveform = self._model.preprocess_audio(binary_data)
        return await self._model.transcribe(
            waveform, timestamps=True, raise_when_trimmed=True
        )
