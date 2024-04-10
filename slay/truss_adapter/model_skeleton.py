import pathlib

import pydantic
from slay import definitions
from truss.templates.shared import secrets_resolver

# Better: in >=3.10 use `TypeAlias`.
UserConfigT = pydantic.BaseModel


class TrussProcessorModel:
    _context: definitions.DeploymentContext[UserConfigT]
    _processor: definitions.ABCProcessor

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata: definitions.TrussMetadata[
            UserConfigT
        ] = definitions.TrussMetadata[UserConfigT].parse_obj(
            config["model_metadata"][definitions.TRUSS_CONFIG_SLAY_KEY]
        )
        self._context = definitions.DeploymentContext[UserConfigT](
            user_config=truss_metadata.user_config,
            processor_to_service=truss_metadata.processor_to_service,
            secrets=secrets,
        )

    # Below illustrated code will be added by code generation.

    # def load(self):
    #     self._processor = {ProcssorCls}(self._context)

    # Sync or async.
    # def predict(self, payload):
    #     return self._processor.{method_name}(payload)
