import importlib
import inspect
import logging
import traceback
from pathlib import Path
from typing import Dict, List

import kfserving
from secrets_resolver import SecretsResolver

MODEL_BASENAME = 'model'


class ModelWrapper(kfserving.KFModel):
    def __init__(self, config: dict):
        super().__init__(MODEL_BASENAME)
        self._config = config
        self._model = None

    def load(self):
        model_module_name = str(Path(self._config['model_class_filename']).with_suffix(''))
        module = importlib.import_module(f"{self._config['model_module_dir']}.{model_module_name}")
        model_class = getattr(module, self._config['model_class_name'])
        model_class_signature = inspect.signature(model_class)
        model_init_params = {}
        if _signature_accepts_keyword_arg(model_class_signature, 'config'):
            model_init_params['config'] = self._config
        if _signature_accepts_keyword_arg(model_class_signature, 'data_dir'):
            model_init_params['data_dir'] = Path('data')
        if _signature_accepts_keyword_arg(model_class_signature, 'secrets'):
            model_init_params['secrets'] = SecretsResolver.get_secrets(self._config)
        self._model = model_class(**model_init_params)
        if hasattr(self._model, 'load'):
            self._model.load()
        self.ready = True

    def preprocess(self, request: Dict) -> Dict:
        if not hasattr(self._model, 'preprocess'):
            return request
        return self._model.preprocess(request)

    def postprocess(self, request: Dict) -> Dict:
        if not hasattr(self._model, 'postprocess'):
            return request
        return self._model.postprocess(request)

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        try:
            return self._model.predict(request)
        except Exception:
            logging.error(traceback.format_exc())
            response['error'] = {'traceback': traceback.format_exc()}
            return response


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False
