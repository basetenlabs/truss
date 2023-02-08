from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_SBERT = "all-MiniLM-L6-v2"


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None
        self._device = None
        self._device_fallback = "cpu"
        self._model_preprocesser = None

    def load(self):
        self._device_predict = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._model = SentenceTransformer(DEFAULT_SBERT, device=self._device_fallback)

    def prepare(self):
        self._model.to(self._device_predict)

    def standby(self):
        self._model.to(self._device_fallback)

    def preprocess(self, request: Dict) -> Dict:
        # Our sentences we like to encode
        request["sentences"] = [
            # "This framework generates embeddings for each input sentence",
            # "Sentences are passed as a list of string.",
            # "The quick brown fox jumps over the lazy dog.",
        ]
        return request

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        sentences = request["sentences"]

        # Sentences are encoded by calling model.encode()
        sentence_embeddings = self._model.encode(sentences)
        result = {"lang": "en", "predictions": []}
        # Print the embeddings
        for sentence, embedding in zip(sentences, sentence_embeddings):
            result["predictions"].append({"sentence": sentence, "embedding": embedding})
        return result
