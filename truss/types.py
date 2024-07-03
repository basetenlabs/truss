from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel

from truss.patch.types import TrussSignature
from truss.templates.control.control.helpers.types import Patch


class ModelFrameworkType(Enum):
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    PYTORCH = "pytorch"
    HUGGINGFACE_TRANSFORMER = "huggingface_transformer"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    MLFLOW = "mlflow"
    CUSTOM = "custom"


@dataclass
class Example:
    name: str
    input: Any

    @staticmethod
    def from_dict(example_dict):
        return Example(
            name=example_dict["name"],
            input=example_dict["input"],
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input": self.input,
        }


@dataclass
class PatchDetails:
    prev_hash: str
    prev_signature: TrussSignature
    next_hash: str
    next_signature: TrussSignature
    patch_ops: List[Patch]

    def to_dict(self):
        return {
            "prev_hash": self.prev_hash,
            "prev_signature": self.prev_signature.to_dict(),
            "next_hash": self.next_hash,
            "next_signature": self.next_signature.to_dict(),
            "patch_ops": [patch_op.to_dict() for patch_op in self.patch_ops],
        }

    def is_empty(self) -> bool:
        # It's possible for prev_hash and next_hash to be different and yet
        # patch_ops to be empty, because certain parts of truss may be ignored.
        return len(self.patch_ops) == 0

    @staticmethod
    def from_dict(patch_details: Dict) -> "PatchDetails":
        return PatchDetails(
            prev_hash=patch_details["prev_hash"],
            prev_signature=TrussSignature.from_dict(patch_details["prev_signature"]),
            next_hash=patch_details["next_hash"],
            next_signature=TrussSignature.from_dict(patch_details["next_signature"]),
            patch_ops=[
                Patch.from_dict(patch_op) for patch_op in patch_details["patch_ops"]
            ],
        )


class PatchRequest(BaseModel):
    """Request to patch running container"""

    hash: str
    prev_hash: str
    patches: List[Patch]

    def to_dict(self):
        return {
            "hash": self.hash,
            "prev_hash": self.prev_hash,
            "patches": [patch.to_dict() for patch in self.patches],
        }

    @staticmethod
    def from_dict(patch_request_dict: Dict):
        current_hash = patch_request_dict["hash"]
        prev_hash = patch_request_dict["prev_hash"]
        patches = patch_request_dict["patches"]
        return PatchRequest(hash=current_hash, prev_hash=prev_hash, patches=patches)
