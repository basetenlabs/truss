import base64
import tempfile

import data_types

import truss_chains as chains
from truss.base import truss_config


def base64_to_wav(base64_string, output_file_path):
    binary_data = base64.b64decode(base64_string)
    with open(output_file_path, "wb") as wav_file:
        wav_file.write(binary_data)
    return output_file_path


@chains.mark_entrypoint
class WhisperModel(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            base_image=chains.CustomImage(
                image="baseten/truss-server-base:3.10-gpu-v0.9.0"
            ),
            apt_requirements=["ffmpeg"],
            pip_requirements=["torch==2.0.1", "openai-whisper==20231106"],
        ),
        compute=chains.Compute(gpu="T4", cpu_count=2, memory="16Gi"),
        assets=chains.Assets(
            external_data=[
                truss_config.ExternalDataItem(
                    local_data_path="weights/large-v3.pt",
                    url="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",  # noqa: E501
                )
            ]
        ),
    )

    def __init__(
        self, context: chains.DeploymentContext = chains.depends_context()
    ) -> None:
        import torch
        import whisper

        self._data_dir = context.data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._data_dir is None:
            raise ValueError("data_dir is required for WhisperModel")
        self.model = whisper.load_model(  # type: ignore [attr-defined]
            self._data_dir / "weights" / "large-v3.pt", self.device
        )

    async def run_remote(
        self, whisper_input: data_types.WhisperInput
    ) -> data_types.WhisperResult:
        import whisper

        with tempfile.NamedTemporaryFile() as fp:
            base64_to_wav(whisper_input.audio_b64, fp.name)
            result = whisper.transcribe(self.model, fp.name, temperature=0)  # type: ignore [attr-defined]
            segments = [
                data_types.WhisperSegment(
                    start_time_sec=r["start"], end_time_sec=r["end"], text=r["text"]
                )
                for r in result["segments"]
            ]

        return data_types.WhisperResult(
            language=whisper.tokenizer.LANGUAGES[result["language"]],  # type: ignore [attr-defined]
            language_code=result["language"],
            segments=segments,
        )
