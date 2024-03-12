import pathlib

from slay import definitions
from truss.templates.shared import secrets_resolver


class Model:
    _context: definitions.Context
    _processor: definitions.ABCProcessor

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata = definitions.TrussMetadata.parse_obj(
            config["model_metadata"]["slay_metadata"]
        )
        self._context = definitions.Context(
            user_config=truss_metadata.user_config,
            stub_cls_to_url=truss_metadata.stub_cls_to_url,
            secrets=secrets,
        )

    # def load(self):
    #     self._processor = {ProcssorCls}(self._context)

    # Sync async.
    # def predict(self, payload):
    #     return self._processor.{method_name}(payload)
