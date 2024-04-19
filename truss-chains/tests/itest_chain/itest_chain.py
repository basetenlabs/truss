import math

import pydantic
import truss_chains as chains
from user_package import shared_chainlet

IMAGE_COMMON = chains.DockerImage(
    pip_requirements_file=chains.make_abs_path_here("requirements.txt")
)


class GenerateData(chains.ChainletBase):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def run(self, length: int) -> str:
        template = "erodfd"
        repetitions = int(math.ceil(length / len(template)))
        return (template * repetitions)[:length]


class DummyUserConfig(pydantic.BaseModel):
    multiplier: int


class TextReplicator(chains.ChainletBase[DummyUserConfig]):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)
    default_user_config = DummyUserConfig(multiplier=2)

    def run(self, data: str) -> str:
        if len(data) > 30:
            raise ValueError(f"This input is too long: {len(data)}.")
        return data * self.user_config.multiplier


class TextToNum(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        replicator: TextReplicator = chains.provide(TextReplicator),
    ) -> None:
        super().__init__(context)
        self._replicator = replicator

    def run(self, data: str) -> int:
        number = 0
        generated_text = self._replicator.run(data)
        for char in generated_text:
            number += ord(char)

        return number


class ItestChain(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        data_generator: GenerateData = chains.provide(GenerateData),
        splitter: shared_chainlet.SplitText = chains.provide(shared_chainlet.SplitText),
        text_to_num: TextToNum = chains.provide(TextToNum),
    ) -> None:
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run(self, length: int, num_partitions: int) -> tuple[int, str, int]:
        data = self._data_generator.run(length)
        text_parts, number = await self._data_splitter.run(
            shared_chainlet.SplitTextInput(
                data=data,
                num_partitions=num_partitions,
                mode=shared_chainlet.Modes.MODE_1,
            ),
            extra_arg=123,
        )
        value = 0
        for part in text_parts.parts:
            value += self._text_to_num.run(part)
        return value, data, number
