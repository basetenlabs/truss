import math

from user_package import shared_chainlet
from user_package.nested_package import io_types

import truss_chains as chains

IMAGE_BASETEN = chains.DockerImage(
    base_image=chains.BasetenImage.PY310,
    pip_requirements_file=chains.make_abs_path_here("requirements.txt"),
)

IMAGE_CUSTOM = chains.DockerImage(
    # TODO: Specifying py path gives a weird error during remote build (works locally)
    #   0.037 /bin/sh: 1: pip: Too many levels of symbolic links
    base_image=chains.CustomImage(
        image="python:3.11-slim"  # , python_executable_path="/usr/local/bin/python"
    ),
    pip_requirements_file=chains.make_abs_path_here("requirements.txt"),
)


class GenerateData(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=IMAGE_BASETEN, name="GENERATE_DATA"
    )

    def run_remote(self, length: int) -> str:
        template = "erodfd"
        repetitions = int(math.ceil(length / len(template)))
        return (template * repetitions)[:length]


class TextReplicator(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_CUSTOM)

    def __init__(self):
        try:
            import pytzdata

            print(f"Could import {pytzdata} is present")
        except ModuleNotFoundError:
            print("Could not import pytzdata is present")
        self.multiplier = 2

    def run_remote(self, data: str) -> str:
        if len(data) > 30:
            raise ValueError(f"This input is too long: {len(data)}.")
        return data * self.multiplier


class SideEffectBase(chains.ChainletBase):
    def __init__(self, context=chains.depends_context()):
        self.ctx = context

    def run_remote(self) -> None:
        print("I'm have no input and no outputs, I just print.")


class SideEffectOnlySubclass(SideEffectBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_CUSTOM)

    def __init__(self, context=chains.depends_context()):
        super().__init__(context=context)

    def run_remote(self) -> None:
        return super().run_remote()


class TextToNum(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_BASETEN)

    def __init__(
        self,
        replicator: TextReplicator = chains.depends(TextReplicator),
        side_effect=chains.depends(SideEffectOnlySubclass),
    ) -> None:
        self._replicator = replicator
        self._side_effect = side_effect

    def run_remote(self, data: str) -> int:
        number = 0
        generated_text = self._replicator.run_remote(data)
        for char in generated_text:
            number += ord(char)

        self._side_effect.run_remote()
        return number


@chains.mark_entrypoint
class ItestChain(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=IMAGE_BASETEN)

    def __init__(
        self,
        data_generator: GenerateData = chains.depends(GenerateData),
        splitter=chains.depends(shared_chainlet.SplitTextFailOnce, retries=2),
        text_to_num: TextToNum = chains.depends(TextToNum),
        context=chains.depends_context(),
    ) -> None:
        self._context = context
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run_remote(
        self,
        length: int,
        num_partitions: int,
        pydantic_default_arg: shared_chainlet.SplitTextOutput = shared_chainlet.SplitTextOutput(
            parts=[], part_lens=[10]
        ),
        simple_default_arg: list[str] = ["a", "b"],
    ) -> tuple[int, str, int, shared_chainlet.SplitTextOutput, list[str]]:
        data = self._data_generator.run_remote(length)
        text_parts, number = await self._data_splitter.run_remote(
            io_types.SplitTextInput(
                data=data,
                num_partitions=num_partitions,
                mode=io_types.Modes.MODE_1,
            ),
            extra_arg=123,
        )
        print(pydantic_default_arg, simple_default_arg)
        value = 0
        for part in text_parts.parts:
            value += self._text_to_num.run_remote(part)
        return (
            value,
            data,
            number,
            pydantic_default_arg,
            simple_default_arg,
        )
