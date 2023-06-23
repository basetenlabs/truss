from abc import ABC, abstractmethod
from pathlib import Path


class BlobBackend(ABC):
    """A blob backend downloads large remote files."""

    @abstractmethod
    def download(self, url: str, download_to: Path):
        raise NotImplementedError()
