from typing import Optional

# flake8: noqa F402
# This location assumes `fde`-repo is checked out at the same level as `truss`-repo.
_LOCAL_WHISPER_LIB = "../../../../fde/whisper-trt/src"
import sys

sys.path.append(_LOCAL_WHISPER_LIB)

import base64

import pydantic
import truss_chains as chains
from huggingface_hub import snapshot_download


# TODO: The I/O types below should actually be taken from `whisper_trt.types`.
#  But that cannot be imported without having `tensorrt_llm` installed.
#  It could be fixed, by making that module importable without any special requirements.
class Segment(pydantic.BaseModel):
    start_time_sec: float
    end_time_sec: float
    text: str
    start: float  # TODO: deprecate, use field with unit (seconds).
    end: float  # TODO: deprecate, use field with unit (seconds).


class WhisperResult(pydantic.BaseModel):
    segments: list[Segment]
    language: Optional[str]
    language_code: Optional[str] = pydantic.Field(
        ...,
        description="IETF language tag, e.g. 'en', see. "
        "https://en.wikipedia.org/wiki/IETF_language_tag.",
    )


class WhisperInput(pydantic.BaseModel):
    audio_b64: str


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

    async def run_remote(self, request: WhisperInput) -> WhisperResult:
        binary_data = base64.b64decode(request.audio_b64.encode("utf-8"))
        waveform = self._model.preprocess_audio(binary_data)
        return await self._model.transcribe(
            waveform, timestamps=True, raise_when_trimmed=True
        )
