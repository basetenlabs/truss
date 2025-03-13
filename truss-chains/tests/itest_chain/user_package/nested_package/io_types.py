import enum

import pydantic


class Modes(str, enum.Enum):
    MODE_0 = "MODE_0"
    MODE_1 = "MODE_1"


class SplitTextInput(pydantic.BaseModel):
    data: str
    num_partitions: int
    mode: Modes


class Item(pydantic.BaseModel):
    number: int
