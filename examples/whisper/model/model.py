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

    def preprocess(self, model_input: Dict) -> Dict:
        resp = requests.get(model_input["url"])
        return {"response": resp.content}

    def postprocess(self, model_output: Dict) -> Dict:
        return model_output

    def predict(self, model_input: Dict) -> Dict:
        with NamedTemporaryFile() as fp:
            fp.write(model_input["response"])
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
