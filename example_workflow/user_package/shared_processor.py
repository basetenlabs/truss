import slay

IMAGE_NUMPY = slay.Image().pip_install("numpy")


class SplitText(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_NUMPY)

    async def split(self, data: str, num_partitions: int) -> list[str]:
        import numpy as np

        parts = np.array_split(np.array(list(data)), 3)
        return ["".join(part) for part in parts]
