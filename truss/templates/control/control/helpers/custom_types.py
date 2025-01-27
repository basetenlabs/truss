from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Type, Union


class PatchType(Enum):
    """Types of console requests sent to Django and passed along to pynode."""

    MODEL_CODE = "model_code"
    PYTHON_REQUIREMENT = "python_requirement"
    SYSTEM_PACKAGE = "system_package"
    CONFIG = "config"
    PACKAGE = "package"
    DATA = "data"
    ENVIRONMENT_VARIABLE = "environment_variable"
    EXTERNAL_DATA = "external_data"


class Action(Enum):
    """Types of console requests sent to Django and passed along to pynode."""

    ADD = "ADD"
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
    content: Optional[str] = None

    def to_dict(self):
        return {"action": self.action.value, "path": self.path, "content": self.content}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return ModelCodePatch(
            action=Action[action_str],
            path=patch_dict["path"],
            content=patch_dict["content"],
        )


@dataclass
class PythonRequirementPatch(PatchBody):
    # For uninstall this should just be the name of the package, but for update
    # should be the full line in the requirements.txt format.
    requirement: str

    def to_dict(self):
        return {"action": self.action.value, "requirement": self.requirement}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return PythonRequirementPatch(
            action=Action[action_str], requirement=patch_dict["requirement"]
        )


@dataclass
class SystemPackagePatch(PatchBody):
    # For uninstall this should just be the name of the package, but for update
    # should be the full line in the requirements.txt format.
    package: str

    def to_dict(self):
        return {"action": self.action.value, "package": self.package}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return SystemPackagePatch(
            action=Action[action_str], package=patch_dict["package"]
        )


@dataclass
class ConfigPatch(PatchBody):
    config: dict
    path: str = "config.yaml"

    def to_dict(self):
        return {"action": self.action.value, "config": self.config, "path": self.path}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return ConfigPatch(
            action=Action[action_str],
            config=patch_dict["config"],
            path=patch_dict["path"],
        )


@dataclass
class DataPatch(PatchBody):
    path: str
    content: Optional[str] = None

    def to_dict(self):
        return {"action": self.action.value, "content": self.content, "path": self.path}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return DataPatch(
            action=Action[action_str],
            content=patch_dict["content"],
            path=patch_dict["path"],
        )


@dataclass
class PackagePatch(PatchBody):
    path: str
    content: Optional[str] = None

    def to_dict(self):
        return {"action": self.action.value, "content": self.content, "path": self.path}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return PackagePatch(
            action=Action[action_str],
            content=patch_dict["content"],
            path=patch_dict["path"],
        )


@dataclass
class EnvVarPatch(PatchBody):
    item: dict

    def to_dict(self):
        return {"action": self.action.value, "item": self.item}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return EnvVarPatch(action=Action[action_str], item=patch_dict["item"])


@dataclass
class ExternalDataPatch(PatchBody):
    item: Dict[str, str]

    def to_dict(self):
        return {"action": self.action.value, "item": self.item}

    @staticmethod
    def from_dict(patch_dict: Dict):
        action_str = patch_dict["action"]
        return ExternalDataPatch(action=Action[action_str], item=patch_dict["item"])


PATCH_BODY_BY_TYPE: Dict[
    PatchType,
    Type[
        Union[
            ModelCodePatch,
            PythonRequirementPatch,
            SystemPackagePatch,
            ConfigPatch,
            DataPatch,
            PackagePatch,
            EnvVarPatch,
            ExternalDataPatch,
        ]
    ],
] = {
    PatchType.MODEL_CODE: ModelCodePatch,
    PatchType.PYTHON_REQUIREMENT: PythonRequirementPatch,
    PatchType.SYSTEM_PACKAGE: SystemPackagePatch,
    PatchType.CONFIG: ConfigPatch,
    PatchType.DATA: DataPatch,
    PatchType.PACKAGE: PackagePatch,
    PatchType.ENVIRONMENT_VARIABLE: EnvVarPatch,
    PatchType.EXTERNAL_DATA: ExternalDataPatch,
}


@dataclass
class Patch:
    """Request to execute code on console."""

    type: PatchType
    body: PatchBody

    def to_dict(self):
        return {"type": self.type.value, "body": self.body.to_dict()}

    @staticmethod
    def from_dict(patch_dict: Dict):
        typ = PatchType(patch_dict["type"])
        body = PATCH_BODY_BY_TYPE[typ].from_dict(patch_dict["body"])
        return Patch(typ, body)
