import pathlib

import slay
from slay import definitions, stub
from truss.server.shared import secrets_resolver
from user_dependencies import IMAGE_COMMON, Parameters

from . import user_stubs


class TextToNum(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        mistral = stub.stub_factory(user_stubs.MistralLLM, context)
        super().__init__(context)
        self._mistral = mistral

    def to_num(self, data: str, params: Parameters) -> int:
        number = 0
        generated_text = self._mistral.llm_gen(data)
        for char in generated_text:
            number += ord(char)

        return number


class Model:
    _context: definitions.Context
    _processor: TextToNum

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata = definitions.TrussMetadata.model_validate(
            config["model_metadata"]["slay_metadata"]
        )
        self._context = definitions.Context(
            user_config=truss_metadata.user_config,
            stub_cls_to_url=truss_metadata.stub_cls_to_url,
            secrets=secrets,
        )

    def load(self) -> None:
        self._processor = TextToNum(context=self._context)

    def predict(self, payload):
        result = self._processor.to_num(
            data=payload["data"], params=Parameters.model_validate(payload["params"])
        )
        return result
