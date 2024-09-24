import os

from transformers import pipeline


def print_files_recursively(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = pipeline(
            "fill-mask",
            model="julien-c/EsperBERTo-small",
            tokenizer="julien-c/EsperBERTo-small",
        )
        print_files_recursively(
            "/root/.cache/huggingface/hub/models--julien-c--EsperBERTo-small/"
        )

    def predict(self, model_input):
        # Run model inference here
        return model_input
