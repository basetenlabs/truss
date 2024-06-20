# flake8: noqa F402
# This location assumes `fde`-repo is checked out at the same level as `truss`-repo.
_LOCAL_WHISPER_LIB = "../../../../fde/whisper-trt/src"
import sys

sys.path.append(_LOCAL_WHISPER_LIB)

import base64

import data_types
import truss_chains as chains


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
    )

    def __init__(
        self,
    ) -> None:
        from whisper_trt import WhisperModel
        self._model = WhisperModel.from_model_name("large-v3", max_queue_time=0.010)

    async def run_remote(
        self, whisper_input: data_types.WhisperInput
    ) -> data_types.WhisperResult:
        binary_data = base64.b64decode(whisper_input.audio_b64.encode("utf-8"))
        waveform = self._model.preprocess_audio(binary_data)
        return await self._model.generate(
            waveform,
            prompt=whisper_input.prompt,
            timestamps=whisper_input.timestamps,
            language=whisper_input.language,
            prefix=whisper_input.prefix,
            max_new_tokens=whisper_input.max_new_tokens,
            task=whisper_input.task,
            raise_when_trimmed=True,
        )
