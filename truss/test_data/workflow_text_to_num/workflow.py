import math

import pydantic
import slay
from user_package import shared_processor

IMAGE_COMMON = slay.DockerImage().pip_requirements_file(
    slay.make_abs_path_here("requirements.txt")
)


class GenerateData(slay.ProcessorBase):

    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)

    def run(self, length: int) -> str:
        template = "erodfd"
        repetitions = int(math.ceil(length / len(template)))
        return (template * repetitions)[:length]


class DummyUserConfig(pydantic.BaseModel):
    multiplier: int


class TextReplicator(slay.ProcessorBase[DummyUserConfig]):

    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)
    default_user_config = DummyUserConfig(multiplier=2)

    def run(self, data: str) -> str:
        if len(data) > 30:
            raise ValueError(f"This input is too long: {len(data)}.")
        return data * self.user_config.multiplier


class TextToNum(slay.ProcessorBase):
    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.DeploymentContext = slay.provide_context(),
        replicator: TextReplicator = slay.provide(TextReplicator),
    ) -> None:
        super().__init__(context)
        self._replicator = replicator

    def run(self, data: str) -> int:
        number = 0
        generated_text = self._replicator.run(data)
        for char in generated_text:
            number += ord(char)

        return number


class Workflow(slay.ProcessorBase):
    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.DeploymentContext = slay.provide_context(),
        data_generator: GenerateData = slay.provide(GenerateData),
        splitter: shared_processor.SplitText = slay.provide(shared_processor.SplitText),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run(self, length: int, num_partitions: int) -> tuple[int, str, int]:
        data = self._data_generator.run(length)
        text_parts, number = await self._data_splitter.run(
            shared_processor.SplitTextInput(
                data=data,
                num_partitions=num_partitions,
                mode=shared_processor.Modes.MODE_1,
            ),
            extra_arg=123,
        )
        value = 0
        for part in text_parts.parts:
            value += self._text_to_num.run(part)
        return value, data, number