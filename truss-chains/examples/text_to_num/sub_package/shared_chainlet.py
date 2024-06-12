import enum
from typing import Tuple

import pydantic

import truss_chains as chains


class Modes(str, enum.Enum):
    MODE_0 = "MODE_0"
    MODE_1 = "MODE_1"


class SplitTextInput(pydantic.BaseModel):
    data: str
    num_partitions: int
    mode: Modes


class SplitTextOutput(pydantic.BaseModel):
    parts: list[str]
    part_lens: list[int]


class SplitText(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(pip_requirements=["numpy"])
    )

    async def run_remote(
        self, inputs: SplitTextInput, extra_arg: int
    ) -> Tuple[SplitTextOutput, int]:
        import numpy as np

        if inputs.mode == Modes.MODE_0:
            print(f"Using mode: `{inputs.mode}`")
        elif inputs.mode == Modes.MODE_1:
            print(f"Using mode: `{inputs.mode}`")
        else:
            raise NotImplementedError(inputs.mode)

        parts_arr = np.array_split(np.array(list(inputs.data)), inputs.num_partitions)
        parts = ["".join(part) for part in parts_arr]
        part_lens = [len(part) for part in parts]
        return SplitTextOutput(parts=parts, part_lens=part_lens), extra_arg
