import slay


class SplitText(slay.ProcessorBase):

    default_config = slay.Config(
        image=slay.Image()
        .pip_requirements_file(slay.make_abs_path_here("../requirements.txt"))
        .pip_requirements(["numpy"])
    )

    async def run(self, data: str, num_partitions: int) -> tuple[list[str], int]:
        import numpy as np

        parts = np.array_split(np.array(list(data)), num_partitions)
        return ["".join(part) for part in parts], 123
