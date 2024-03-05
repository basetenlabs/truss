import slay
from slay import stub
from user_dependencies import IMAGE_COMMON, Parameters

from . import dependencies


class TextToNum(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        mistral = stub.stub_factory(dependencies.MistralLLM, context)
        super().__init__(context)
        self._mistral = mistral

    def to_num(self, data: str, params: Parameters) -> int:
        number = 0
        generated_text = self._mistral.llm_gen(data)
        for char in generated_text:
            number += ord(char)

        return number
