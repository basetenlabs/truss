from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum


class PatchType(Enum):
    """Types of console requests sent to Django and passed along to pynode."""

    MODEL_CODE = "model_code"


class Action(Enum):
    """Types of console requests sent to Django and passed along to pynode."""

    UPDATE = "UPDATE"
    REMOVE = "REMOVE"


@dataclass
class PatchBody:
    """Marker class"""

    action: Action

    @abstractmethod
    def to_dict(self):
        pass


@dataclass
class ModelCodePatch(PatchBody):
    path: str  # Relative to model module directory
    content: str = None

    def to_dict(self):
        return {
            "action": self.action.value,
            "path": self.path,
            "content": self.content,
        }

    @staticmethod
    def from_dict(patch_dict: dict):
        action_str = patch_dict["action"]
        return ModelCodePatch(
            action=Action[action_str],
            path=patch_dict["path"],
            content=patch_dict["content"],
        )


PATCH_BODY_BY_TYPE = {
    PatchType.MODEL_CODE: ModelCodePatch,
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
        typ = PatchType(patch_dict["type"])
        body = PATCH_BODY_BY_TYPE[typ].from_dict(patch_dict["body"])
        return Patch(typ, body)
