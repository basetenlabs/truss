from dataclasses import dataclass
from enum import Enum


class PatchType(Enum):
    """Types of console requests sent to Django and passed along to pynode."""

    MODEL_CODE = "model_code"


class Action(Enum):
    """Types of console requests sent to Django and passed along to pynode."""

    UPDATE = "UPDATE"
    REMOVE = "REMOVE"


class PatchBody:
    """Marker class"""

    pass


@dataclass
class ModelCodePatch:
    action: Action
    filepath: str
    content: str

    def to_dict(self):
        return {
            "action": self.action.value,
            "filepath": self.filepath,
            "content": self.content,
        }

    @staticmethod
    def from_dict(patch_dict: dict):
        action_str = patch_dict["action"]
        return ModelCodePatch(
            action=Action[action_str],
            filepath=patch_dict["filepath"],
            content=patch_dict["content"],
        )


PATCH_BODY_BY_TYPE = {
    PatchType.MODEL_CODE.value: ModelCodePatch,
}


@dataclass
class Patch:
    """Request to execute code on console."""

    type: PatchType
    body: PatchBody

    def to_dict(self):
        return {
            "type": self.type.value,
            "body": self.body.to_dict(),
        }

    @staticmethod
    def from_dict(patch_dict: dict):
        typ = patch_dict["type"]
        body = PatchType[typ].from_dict(patch_dict["body"])
        return Patch(typ, body)
