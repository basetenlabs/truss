import math

import pydantic
import truss_chains as chains
from user_package import shared_chainlet
from user_package.nested_package import io_types

IMAGE_COMMON = chains.DockerImage(
    pip_requirements_file=chains.make_abs_path_here("requirements.txt")
)


class GenerateData(chains.ChainletBase):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def run_remote(self, length: int) -> str:
        template = "erodfd"
        repetitions = int(math.ceil(length / len(template)))
        return (template * repetitions)[:length]


class DummyUserConfig(pydantic.BaseModel):
    multiplier: int


class TextReplicator(chains.ChainletBase):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)
    default_user_config = DummyUserConfig(multiplier=2)

    def __init__(self, context=chains.depends_context()):
        self.user_config = context.user_config

    def run_remote(self, data: str) -> str:
        if len(data) > 30:
            raise ValueError(f"This input is too long: {len(data)}.")
        return data * self.user_config.multiplier


class TextToNum(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        replicator: TextReplicator = chains.depends(TextReplicator),
    ) -> None:
        self._replicator = replicator

    def run_remote(self, data: str) -> int:
        number = 0
        generated_text = self._replicator.run_remote(data)
        for char in generated_text:
            number += ord(char)

        return number


class ItestChain(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        data_generator: GenerateData = chains.depends(GenerateData),
        splitter=chains.depends(shared_chainlet.SplitTextFailOnce, retries=2),
        text_to_num: TextToNum = chains.depends(TextToNum),
    ) -> None:
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run_remote(
        self, length: int, num_partitions: int
    ) -> tuple[int, str, int]:
        data = self._data_generator.run_remote(length)
        text_parts, number = await self._data_splitter.run_remote(
            io_types.SplitTextInput(
                data=data,
                num_partitions=num_partitions,
                mode=io_types.Modes.MODE_1,
            ),
            extra_arg=123,
        )
        value = 0
        for part in text_parts.parts:
            value += self._text_to_num.run_remote(part)
        return value, data, number
