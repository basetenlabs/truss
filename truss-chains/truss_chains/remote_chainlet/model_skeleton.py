import pathlib
from typing import Any, Optional

from truss.templates.shared import lazy_data_resolver, secrets_resolver
from truss_chains import private_types, public_types
from truss_chains.remote_chainlet import utils


class TrussChainletModel:
    _context: public_types.DeploymentContext
    _chainlet: private_types.ABCChainlet
    # Populated only for chainlets derived from ``CustomEngineBuilderLLMChainlet``;
    # ``None`` for regular chainlets. The value is the ``trt_llm`` dict the truss
    # server injects when ``config.trt_llm`` is set and the model class declares a
    # ``trt_llm`` kwarg (see ``truss/trt_llm/validation.py``).
    _trt_llm: Optional[Any]

    def __init__(
        self,
        config: dict,
        data_dir: pathlib.Path,
        secrets: secrets_resolver.Secrets,
        lazy_data_resolver: lazy_data_resolver.LazyDataResolverV2,
        # TODO: Remove the default value once all truss versions are synced up.
        environment: Optional[dict] = None,
        # Optional — only supplied by the truss server when the chainlet was
        # generated from a ``CustomEngineBuilderLLMChainlet`` (i.e. the truss
        # config has ``trt_llm`` set and the model-class declares this kwarg).
        trt_llm: Optional[Any] = None,
    ) -> None:
        truss_metadata: private_types.TrussMetadata = (
            private_types.TrussMetadata.model_validate(
                config["model_metadata"][private_types.TRUSS_CONFIG_CHAINS_KEY]
            )
        )
        deployment_environment: Optional[public_types.Environment] = (
            public_types.Environment.model_validate(environment)
            if environment
            else None
        )
        chainlet_to_deployed_service = utils.populate_chainlet_service_predict_urls(
            truss_metadata.chainlet_to_service
        )

        self._context = public_types.DeploymentContext(
            chainlet_to_service=chainlet_to_deployed_service,
            secrets=secrets,
            data_dir=data_dir,
            environment=deployment_environment,
        )
        self._trt_llm = trt_llm
        lazy_data_resolver.block_until_download_complete()

    # Below illustrated code will be added by code generation.

    # def load(self) -> None:
    #     logging.info(f"Loading Chainlet `TextToNum`.")
    #     self._chainlet = itest_chain.TextToNum(
    #         replicator=stub.factory(TextReplicator, self._context),
    #         side_effect=stub.factory(SideEffectOnlySubclass, self._context),
    #     )
    #
    # If chainlet implements is_healthy:
    # def is_healthy(self) -> Optional[bool]:
    #     if hasattr(self, "_chainlet"):
    #         return self._chainlet.is_healthy()
    #
    # def predict(
    #     self, inputs: TextToNumInput, request: starlette.requests.Request
    # ) -> TextToNumOutput:
    #     with utils.predict_context(request.headers):
    #         result = self._chainlet.run_remote(**utils.pydantic_set_field_dict(inputs))
    #     return TextToNumOutput(result)
