from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_MAX_LENGTH = 512
MIN_MAX_LENGTH = 64


SUPPORTED_MODEL_PARAMS = {
    'max_length',
    'min_length',
    'do_sample',
    'early_stopping',
    'num_beams',
    'temperature',
    'top_k',
    'top_p',
    'repetition_penalty',
    'length_penalty',
    'encoder_no_repeat_ngram_size',
    'num_return_sequences',
    'max_time',
    'num_beam_groups',
    'diversity_penalty',
    'remove_invalid_values',
}


def _process_request_into_model_call(request_dict):
    model_parameters = {}
    for param in SUPPORTED_MODEL_PARAMS:
        if param in request_dict:
            model_parameters[param] = request_dict[param]
            if param == 'max_length' and model_parameters[param] > MAX_MAX_LENGTH:
                model_parameters[param] = MAX_MAX_LENGTH
            if param == 'min_length' and model_parameters[param] > MIN_MAX_LENGTH:
                model_parameters[param] = MIN_MAX_LENGTH
    return model_parameters


class GPTJTransformerModel(object):
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs['data_dir']
        self._config = kwargs['config']
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self._model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16).to(self.device)
        self.ready = True

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        outputs = []
        inputs = request['inputs'] # noqa
        # We expect a list of queries
        with torch.no_grad():
            for query in inputs:
                prompt = query['prompt']
                output = self._tokenizer([prompt], return_tensors="pt", return_attention_mask=True)
                attention_mask = output.attention_mask
                input_ids = output.input_ids
                params = _process_request_into_model_call(request)
                gen_tokens = self._model.generate(
                    input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    **params,
                )
                gen_text = self._tokenizer.batch_decode(gen_tokens)[0]
                outputs.append(gen_text)

        # Invoke model and calculate predictions here.
        response['predictions'] = outputs
        return response
