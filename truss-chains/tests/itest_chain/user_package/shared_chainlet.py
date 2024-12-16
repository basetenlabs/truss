from typing import List, Tuple  # Cover testing of antique type annotations.

import pydantic

import truss_chains as chains

from .nested_package import io_types  # Cover testing of relative import resolution.


class SplitTextOutput(pydantic.BaseModel):
    parts: List[str]
    part_lens: List[int]


class SplitTextFailOnce(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements_file=chains.make_abs_path_here("../requirements.txt"),
            pip_requirements=["numpy"],
        )
    )

    def __init__(self):
        self._count = 0

    async def run_remote(
        self,
        inputs: io_types.SplitTextInput,
        extra_arg: int,
        list_arg: list[io_types.Item],
    ) -> Tuple[SplitTextOutput, int, list[io_types.Item]]:
        import numpy as np

        print(extra_arg)
        print(list_arg)

        self._count += 1
        if self._count == 1:
            raise ValueError("Haha this is a fake error.")

        if inputs.mode == io_types.Modes.MODE_0:
            print(f"Using mode: `{inputs.mode}`")
        elif inputs.mode == io_types.Modes.MODE_1:
            print(f"Using mode: `{inputs.mode}`")
        else:
            raise NotImplementedError(inputs.mode)

        parts_arr = np.array_split(np.array(list(inputs.data)), inputs.num_partitions)
        parts = ["".join(part) for part in parts_arr]
        part_lens = [len(part) for part in parts]
        return SplitTextOutput(parts=parts, part_lens=part_lens), 123, list_arg
