from dataclasses import dataclass
from enum import Enum
from typing import Any, List

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

    @staticmethod
    def from_dict(patch_details: dict) -> "PatchDetails":
        return PatchDetails(
            prev_hash=patch_details["prev_hash"],
            prev_signature=TrussSignature.from_dict(patch_details["prev_signature"]),
            next_hash=patch_details["next_hash"],
            next_signature=TrussSignature.from_dict(patch_details["next_signature"]),
            patch_ops=[
                Patch.from_dict(patch_op) for patch_op in patch_details["patch_ops"]
            ],
        )
