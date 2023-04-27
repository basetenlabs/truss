from typing import Dict, List

from transformers import T5ForConditionalGeneration, T5Tokenizer


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs.get("secrets")
        self._tokenizer = None
        self._model = None

    def load(self):
        self._tokenizer = T5Tokenizer.from_pretrained(
            "google/flan-t5-xl", use_auth_token=self._secrets["hf_access_token"]
        )
        self._model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xl",
            device_map="auto",
            use_auth_token=self._secrets["hf_access_token"],
        )

    def predict(self, request: Dict) -> Dict[str, List]:
        try:
            decoded_output = []
            prompt = request.pop("prompt")
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(
                "cuda"
            )
            outputs = self._model.generate(input_ids, **request)
            for beam in outputs:
                decoded_output.append(
                    self._tokenizer.decode(beam, skip_special_tokens=True)
                )
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": decoded_output, "message": None}
