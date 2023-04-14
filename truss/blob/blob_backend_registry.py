from typing import Dict

from truss.blob.blob_backend import BlobBackend
from truss.blob.http_public_blob_backend import HttpPublic
from truss.constants import HTTP_PUBLIC_BLOB_BACKEND


class _BlobBackendRegistry:
    def __init__(self) -> None:
        self._backends: Dict[str, BlobBackend] = {}
        # Register default backend
        self._backends[HTTP_PUBLIC_BLOB_BACKEND] = HttpPublic()

    def register_backend(self, name: str, backend: BlobBackend):
        self._backends[name] = backend

    def get_backend(self, name: str):
        if name not in self._backends:
            raise ValueError(f"Backend {name} is not registered.")
        return self._backends[name]


BLOB_BACKEND_REGISTRY = _BlobBackendRegistry()
