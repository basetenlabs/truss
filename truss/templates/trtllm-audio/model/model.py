import base64

import requests
import sigint_patch

# We patch TensorRT-LLM v0.11 to fix an issue where they set a SIGINT handler
# which breaks because we do not use the main thread.
sigint_patch.patch()
from whisper_trt import WhisperModel
from whisper_trt.custom_types import WhisperResult


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        self._model = WhisperModel(str(self._data_dir), max_queue_time=0.050)

    def preprocess(self, request: dict):
        audio_base64 = request.get("audio")
        audio_url = request.get("url")

        if audio_base64 and audio_url:
            return {
                "error": "Only a base64 audio file OR a URL can be passed to the API, not both of them."
            }
        if not audio_base64 and not audio_url:
            return {
                "error": "Please provide either an audio file in base64 string format or a URL to an audio file."
            }

        binary_data = None

        if audio_base64:
            binary_data = base64.b64decode(audio_base64.encode("utf-8"))
        elif audio_url:
            resp = requests.get(audio_url)
            binary_data = resp.content
        return binary_data, request

    async def predict(self, preprocessed_request) -> WhisperResult:
        binary_data, request = preprocessed_request
        waveform = self._model.preprocess_audio(binary_data)
        return await self._model.transcribe(
            waveform, language="english", timestamps=True
        )
