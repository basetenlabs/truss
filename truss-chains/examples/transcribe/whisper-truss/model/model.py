import base64

import pydantic
from huggingface_hub import snapshot_download
from whisper_trt import WhisperModel
from whisper_trt.types import WhisperResult


class WhisperInput(pydantic.BaseModel):
    audio_b64: str


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        snapshot_download(
            repo_id="baseten/whisper_trt_large-v3_A10G_i224_o512_bs8_bw5",
            local_dir=self._data_dir,
            allow_patterns=["**"],
            token=self._secrets["hf_access_token"],
        )
        self._model = WhisperModel(str(self._data_dir), max_queue_time=0.050)

    async def predict(self, request: WhisperInput) -> WhisperResult:
        binary_data = base64.b64decode(request.audio_b64.encode("utf-8"))
        waveform = self._model.preprocess_audio(binary_data)
        return await self._model.transcribe(
            waveform, timestamps=True, raise_when_trimmed=True
        )
