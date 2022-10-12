import tempfile
from typing import Dict, List

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
        request["detected_lang"] = f"Detected Language: {request['language']}"
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        fp = tempfile.NamedTemporaryFile()
        fp.write(request["response"])
        result = whisper.transcribe(
            self._model,
            fp.name,
            temperature=0,
            best_of=5,
            beam_size=5,
        )
        fp.close()
        return {"text": result["text"], "language": whisper.tokenizer.LANGUAGES[result["language"]]}
