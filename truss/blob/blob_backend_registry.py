from typing import Dict

from truss.blob.blob_backend import BlobBackend


class BlobBackendRegistry:
    backends: Dict[str, BlobBackend]

    @staticmethod
    def register_backend(name: str, backend: BlobBackend):
        BlobBackendRegistry.backends[name] = backend

    @staticmethod
    def get_backend(name: str):
        if name not in BlobBackendRegistry.backends:
            raise ValueError(f"Backend {name} is not registered.")
        return BlobBackendRegistry.backends[name]
