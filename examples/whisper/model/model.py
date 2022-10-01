import os
from typing import Dict, List

import requests
import torch
import whisper


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def load(self):
        self._model = whisper.load_model("small", self.device)
        return

    def preprocess(self, request: Dict) -> Dict:
        resp = requests.get(request['url'])
        tmp_file = '/tmp/tmp.mp3'
        with open(tmp_file, 'wb') as f:
            f.write(resp.content)
        request['file_path'] = tmp_file
        return request


    def postprocess(self, request: Dict) -> Dict:
        request['detected_lang'] = f"Detected Language: {request['language']}"
        os.remove(request['file_path'])
        return request


    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        result = whisper.transcribe(self._model,
                    request['file_path'],
                    temperature=0,
                    best_of=5,
                    beam_size=5
                    )
        response['file_path'] = request['file_path']
        response['text'] = result['text']
        response['language'] = whisper.tokenizer.LANGUAGES[result['language']]
        return response
