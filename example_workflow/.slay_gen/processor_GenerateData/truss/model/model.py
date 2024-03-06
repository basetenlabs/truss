import pathlib
import random
import string

import slay
from slay import definitions
from truss.server.shared import secrets_resolver
from user_dependencies import IMAGE_COMMON, Parameters


class GenerateData(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_COMMON)

    def gen_data(self, params: Parameters) -> str:
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=params.length)
        )


class Model:
    _context: definitions.Context
    _processor: GenerateData

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata = definitions.TrussMetadata.model_validate(
            config["slay_metadata"]
        )
        self._context = definitions.Context(
            user_config=truss_metadata.user_config,
            stub_cls_to_url=truss_metadata.stub_cls_to_url,
            secrets=secrets,
        )

    def load(self) -> None:
        self._processor = GenerateData(self._context)

    def predict(self, payload):
        return self._processor.gen_data(payload)
