import tempfile
from typing import Dict, List

import torch
import whisper
from pytube import YouTube


class Model:
    def __init__(self, **kwargs) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def load(self):
        self._model = whisper.load_model("medium", self._device)
        return

    def preprocess(self, request: Dict) -> Dict:
        return request

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        media_url = request["url"]
        yt = YouTube(media_url)
        itag = yt.streams.filter(only_audio=True)[0].itag
        stream = yt.streams.get_by_itag(itag)
        with tempfile.TemporaryDirectory() as temp_dir_name:
            audio_path = f"{temp_dir_name}/audio"
            with open(audio_path, "wb"):
                stream.download(filename=audio_path)
                result = whisper.transcribe(
                    self._model,
                    audio_path,
                    beam_size=5,
                )
        return {
            "language": whisper.tokenizer.LANGUAGES[result["language"]],
            "text": result["text"],
        }
