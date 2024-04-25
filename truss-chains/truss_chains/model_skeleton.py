import pathlib

import pydantic
from truss.templates.shared import secrets_resolver
from truss_chains import definitions

# Better: in >=3.10 use `TypeAlias`.
UserConfigT = pydantic.BaseModel


class TrussChainletModel:
    _context: definitions.DeploymentContext[UserConfigT]
    _chainlet: definitions.ABCChainlet

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata: definitions.TrussMetadata[
            UserConfigT
        ] = definitions.TrussMetadata[UserConfigT].parse_obj(
            config["model_metadata"][definitions.TRUSS_CONFIG_CHAINS_KEY]
        )
        self._context = definitions.DeploymentContext[UserConfigT](
            user_config=truss_metadata.user_config,
            chainlet_to_service=truss_metadata.chainlet_to_service,
            secrets=secrets,
            data_dir=data_dir,
        )

    # Below illustrated code will be added by code generation.

    # def load(self):
    #     self._chainlet = {ChainletCls}(self._context)

    # Sync or async.
    # def predict(self, payload):
    #     return self._chainlet.{method_name}(payload)
