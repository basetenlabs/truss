import slay

IMAGE_NUMPY = (
    slay.Image()
    .pip_requirements_file(
        "/home/marius-baseten/workbench/truss/example_workflow_mvp/requirements.txt"
    )
    .pip_requirements(["numpy"])
)


class SplitText(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_NUMPY)

    async def run(self, data: str, num_partitions: int) -> tuple[list[str], int]:
        import numpy as np

        parts = np.array_split(np.array(list(data)), 3)
        return ["".join(part) for part in parts], 123
