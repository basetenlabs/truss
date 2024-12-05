import pathlib
from typing import Optional

from truss.templates.shared import secrets_resolver

from truss_chains import definitions
from truss_chains.remote_chainlet import utils


class TrussChainletModel:
    _context: definitions.DeploymentContext
    _chainlet: definitions.ABCChainlet

    def __init__(
        self,
        config: dict,
        data_dir: pathlib.Path,
        secrets: secrets_resolver.Secrets,
        # TODO: Remove the default value once all truss versions are synced up.
        environment: Optional[dict] = None,
    ) -> None:
        truss_metadata: definitions.TrussMetadata = (
            definitions.TrussMetadata.model_validate(
                config["model_metadata"][definitions.TRUSS_CONFIG_CHAINS_KEY]
            )
        )
        deployment_environment: Optional[definitions.Environment] = (
            definitions.Environment.model_validate(environment) if environment else None
        )
        chainlet_to_deployed_service = utils.populate_chainlet_service_predict_urls(
            truss_metadata.chainlet_to_service
        )

        self._context = definitions.DeploymentContext(
            chainlet_to_service=chainlet_to_deployed_service,
            secrets=secrets,
            data_dir=data_dir,
            environment=deployment_environment,
        )

    # Below illustrated code will be added by code generation.

    # def load(self) -> None:
    #     logging.info(f"Loading Chainlet `TextToNum`.")
    #     self._chainlet = itest_chain.TextToNum(
    #         replicator=stub.factory(TextReplicator, self._context),
    #         side_effect=stub.factory(SideEffectOnlySubclass, self._context),
    #     )
    #
    # def predict(
    #     self, inputs: TextToNumInput, request: starlette.requests.Request
    # ) -> TextToNumOutput:
    #     with stub.trace_parent(request), utils.exception_to_http_error(
    #         include_stack=True, chainlet_name="TextToNum"
    #     ):
    #         result = self._chainlet.run_remote(**utils.pydantic_set_field_dict(inputs))
    #     return TextToNumOutput(result)
