from abc import ABC, abstractmethod
from pathlib import Path


class TrussContext(ABC):
    """Marker class for Truss context.

    A model is represented in a standard form called Truss. Truss contexts
    perform a certain operation on this Truss. Some examples are: running the
    model directly, building an image and, deploying to baseten.
    """

    @staticmethod
    @abstractmethod
    def run(truss_dir: Path):
        pass
