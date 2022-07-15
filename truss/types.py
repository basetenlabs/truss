from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelFrameworkType(Enum):
    SKLEARN = 'sklearn'
    TENSORFLOW = 'tensorflow'
    KERAS = 'keras'
    PYTORCH = 'pytorch'
    HUGGINGFACE_TRANSFORMER = 'huggingface_transformer'
    XGBOOST = 'xgboost'
    LIGHTGBM = 'lightgbm'
    CUSTOM = 'custom'


@dataclass
class Example:
    name: str
    input: Any

    @staticmethod
    def from_dict(example_dict):
        return Example(
            name=example_dict['name'],
            input=example_dict['input'],
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'input': self.input,
        }
