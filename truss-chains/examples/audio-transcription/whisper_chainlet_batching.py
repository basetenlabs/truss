# flake8: noqa F402
# This location assumes `fde`-repo is checked out at the same level as `truss`-repo.
_LOCAL_WHISPER_LIB = "/home/marius-baseten/workbench/fde/whisper-trt/src"

import base64

import data_types

import truss_chains as chains


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
        compute=chains.Compute(gpu="L4", predict_concurrency=128),
    )

    def __init__(
        self,
    ) -> None:
        from whisper_trt import WhisperModel

        # Default to 50ms batching delay. Should be tuned for throughput requirements.
        self._model = WhisperModel.from_model_name("large-v3", max_queue_time=0.050)

    async def run_remote(
        self, whisper_input: data_types.WhisperInput
    ) -> data_types.WhisperResult:
        binary_data = base64.b64decode(whisper_input.audio_b64.encode("utf-8"))
        waveform = self._model.preprocess_audio(binary_data)
        # TODO: consider splitting out types from whisper-trt into their own package to eliminate all this ser/deser.
        return data_types.WhisperResult(
            **(
                await self._model.generate(
                    waveform,
                    prompt=None,
                    language=None,
                    prefix=None,
                    max_new_tokens=512,
                    task="transcribe",
                    raise_when_trimmed=True,
                )
            ).model_dump()
        )


if __name__ == "__main__":
    import asyncio
    import json
    import os

    input_audio_file = "/workspace/fde/whisper-trt/assets/1221-135766-0002.wav"

    with open(input_audio_file, "rb") as audio_file:
        whisper_input = data_types.WhisperInput(
            audio_b64=base64.b64encode(audio_file.read()).decode("utf-8")
        )

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):
        whisper_model = WhisperModel()

        result_ = asyncio.run(whisper_model.run_remote(whisper_input))

        print(json.dumps(result_.dict(), indent=4))
