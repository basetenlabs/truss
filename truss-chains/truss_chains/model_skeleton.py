import pathlib
from typing import Optional

from truss.templates.shared import secrets_resolver

from truss_chains import definitions
from truss_chains.utils import override_chainlet_to_service_metadata


class TrussChainletModel:
    _context: definitions.DeploymentContext
    _chainlet: definitions.ABCChainlet

    def __init__(
        self,
        config: dict,
        data_dir: pathlib.Path,
        secrets: secrets_resolver.Secrets,
        environment: Optional[
            dict
        ] = None,  # TODO: Remove the default value once all truss versions are synced up.
    ) -> None:
        truss_metadata: definitions.TrussMetadata = (
            definitions.TrussMetadata.model_validate(
                config["model_metadata"][definitions.TRUSS_CONFIG_CHAINS_KEY]
            )
        )
        deployment_environment: Optional[definitions.Environment] = (
            definitions.Environment.model_validate(environment) if environment else None
        )
        override_chainlet_to_service_metadata(truss_metadata.chainlet_to_service)

        self._context = definitions.DeploymentContext(
            chainlet_to_service=truss_metadata.chainlet_to_service,
            secrets=secrets,
            data_dir=data_dir,
            environment=deployment_environment,
        )

    # Below illustrated code will be added by code generation.

    # def load(self) -> None:
    #     logging.info(f"Loading Chainlet `TextToNum`.")
    #     self._chainlet = main.TextToNum(
    #       mistral=stub.factory(MistralLLM, self._context))
    #
    # def predict(self, inputs: TextToNumInput) -> TextToNumOutput:
    #     with utils.exception_to_http_error(
    #         include_stack=True, chainlet_name="TextToNum"
    #     ):
    #         result = self._chainlet.run_remote(data=inputs.data)
    #     return TextToNumOutput((result,))
