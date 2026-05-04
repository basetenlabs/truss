"""A regular Truss model that participates in a chain by reading sibling URLs
through the public ``truss_chains.runtime`` API.

Notice: no `ChainletBase`, no `run_remote`, no `truss_chains` framework imports
beyond the public runtime module. This Truss could be reused outside any chain
context — its dependence on chain siblings is conditional via
``runtime.list_services()``.
"""

from truss_chains.runtime import get_service, list_services


class Model:
    def __init__(self, **kwargs) -> None:
        self._diarizer_url = None
        self._auth_headers = None

    def load(self) -> None:
        # Conditional sibling discovery: if running inside a chain, pick up
        # the Diarizer URL; if standalone, fall back to no-op behavior.
        siblings = list_services()
        if "Diarizer" in siblings:
            desc = get_service("Diarizer")
            # Caller picks the URL flavor it wants — predict_url for public
            # routing, internal_url for cluster-local.
            self._diarizer_url = desc.target_url
            # In a real Truss, source the api_key from the standard Truss
            # secrets API. For this demo we leave it as a placeholder.
            self._auth_headers = desc.with_auth_headers(api_key="<from-secrets>")

    def predict(self, request: dict) -> dict:
        return {"diarizer_url": self._diarizer_url, "auth_headers": self._auth_headers}
