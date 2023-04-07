import random
from typing import Dict, List

from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        self._model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xl", device_map="auto"
        )

    def preprocess(self, request: dict):
        if "bad_words" in request:
            bad_words = request.pop("bad_words")
            # bad_words must be a list of strings, not one string
            bad_word_ids = self._tokenizer(
                bad_words, add_prefix_space=True, add_special_tokens=False
            ).input_ids
            request["bad_words_ids"] = bad_word_ids
        if "seed" in request:
            set_seed(request.pop("seed"))
        else:
            set_seed(random.randint(0, 4294967294))
        return request

    def assert_free_tier_limits(self, request: dict):
        if request.get("max_new_tokens", 0) > 100:
            raise ValueError(
                "max_tokens / max_new_tokens must be less than 101 on free tier"
            )
        if request.get("num_beams", 0) > 4:
            raise ValueError("num_beams must be less than 5 on free tier")
        if request.get("num_beam_groups", 0) > 4:
            raise ValueError("num_beam_groups must be less than 5 on free tier")

    def predict(self, request: Dict) -> Dict[str, List]:
        try:
            self.assert_free_tier_limits(request)
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
