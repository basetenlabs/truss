"""Plain ``Model`` that optionally resolves a sibling via ``TrussHandle``.

Uses only the public runtime API—no Chainlet framework. On ``MissingDependencyError``,
``.load()`` leaves sibling fields unset.
"""

from truss_chains.public_types import MissingDependencyError
from truss_chains.remote_chainlet.truss_chainlet import TrussHandle


class Model:
    def __init__(self, **kwargs) -> None:
        self._diarizer_url = None
        self._auth_headers = None

    def load(self) -> None:
        # Optional Diarizer sibling; skip if not in a chain.
        try:
            diarizer = TrussHandle("Diarizer")
        except MissingDependencyError:
            return
        # Fixture uses a literal api_key; deployed Truss reads secrets automatically.
        self._diarizer_url, self._auth_headers = diarizer.http_call_args(
            prefer_internal=True, api_key="<from-secrets>"
        )

    def predict(self, request: dict) -> dict:
        return {"diarizer_url": self._diarizer_url, "auth_headers": self._auth_headers}
