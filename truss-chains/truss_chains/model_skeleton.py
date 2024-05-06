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

    # def load(self) -> None:
    #     logging.info(f"Loading Chainlet `SplitText`.")
    #     self._chainlet = shared_chainlet.SplitText(context=self._context)

    # async def predict(self, payload):
    #     with utils.exception_to_http_error(
    #         include_stack=True, chainlet_name="SplitText"
    #     ):
    #         result = await self._chainlet.run(
    #             inputs=shared_chainlet.SplitTextInput.parse_obj(payload["inputs"]),
    #             extra_arg=payload["extra_arg"],
    #         )
    #     return result[0].dict(), result[1]
