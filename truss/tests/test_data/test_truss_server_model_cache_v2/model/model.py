from transformers import pipeline


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        print("loading model from /app/model_cache/julien_c_esper")
        self._model = pipeline(
            "fill-mask",
            model="/app/model_cache/julien_c_esper",
            tokenizer="/app/model_cache/julien_c_esper",
        )

    def predict(self, model_input):
        # Run model inference here
        return model_input
