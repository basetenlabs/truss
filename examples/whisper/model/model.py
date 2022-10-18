from tempfile import NamedTemporaryFile
from typing import Dict

import requests
import torch
import whisper


class Model:
    def __init__(self, **kwargs) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def load(self):
        self._model = whisper.load_model("small", self.device)
        return

    def preprocess(self, request: Dict) -> Dict:
        resp = requests.get(request["url"])
        return {"response": resp.content}

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict:
        with NamedTemporaryFile() as fp:
            fp.write(request["response"])
            result = whisper.transcribe(
                self._model,
                fp.name,
                temperature=0,
                best_of=5,
                beam_size=5,
            )
            segments = [
                {"start": r["start"], "end": r["end"], "text": r["text"]}
                for r in result["segments"]
            ]
        return {
            "language": whisper.tokenizer.LANGUAGES[result["language"]],
            "segments": segments,
            "text": result["text"],
        }
