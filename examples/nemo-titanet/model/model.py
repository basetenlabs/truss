import warnings
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import nemo.collections.asr as nemo_asr
import requests

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )

    def preprocess(self, model_input: Dict) -> Dict:
        resp = requests.get(model_input["url"])
        return {"response": resp.content}

    def predict(self, model_input: Dict) -> Any:
        with NamedTemporaryFile() as fp:
            fp.write(model_input["response"])
            embs = self._model.get_embedding(fp.name)
            return embs.tolist()
