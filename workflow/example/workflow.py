import logging
import random
import string

import pydantic
import slay


class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4
    num_replications: int = 3


class GenerateData(slay.BaseProcessor):

    default_config = slay.Config(name="MyDataGenerator")

    def gen_data(self, params: Parameters) -> str:
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=params.length)
        )


class SplitData(slay.BaseProcessor):
    def split(self, data: str, params: Parameters) -> list[str]:
        num_partitions = params.num_partitions
        part_length = len(data) // num_partitions
        text_parts = [
            data[i * part_length : (i + 1) * part_length]
            + (data[num_partitions * part_length :] if i == num_partitions - 1 else "")
            for i in range(num_partitions)
        ]
        return text_parts


class TextReplicator(slay.BaseProcessor):
    def replicate(self, data: str, params: Parameters) -> str:
        return str(data * params.num_replications)


class TextToNum(slay.BaseProcessor):

    _replicator: TextReplicator

    def __init__(
        self,
        config: slay.Config,
        replicator: TextReplicator = slay.provide(TextReplicator),
    ) -> None:
        super().__init__(config)
        self._replicator = replicator

    def to_num(self, data: str, params: Parameters) -> int:
        number = 0
        replicator_result = self._replicator.replicate(data, params)
        for char in replicator_result:
            number += ord(char)

        return number


class Workflow(slay.BaseProcessor):
    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
        data_generator: GenerateData = slay.provide(GenerateData),
        data_splitter: SplitData = slay.provide(SplitData),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(config)
        self._data_generator = data_generator
        self._data_splitter = data_splitter
        self._text_to_num = text_to_num

    def run(self, params: Parameters) -> int:
        data = self._data_generator.gen_data(params)
        text_parts = self._data_splitter.split(data, params)
        value = 0
        for part in text_parts:
            value += self._text_to_num.to_num(part, params)
        return value


if __name__ == "__main__":
    log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    date_format = "%m%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)

    # Local test or dev execution - context manager makes sure local processors
    # are instantiated and injected.
    # with slay.run_local():
    #     wf = Workflow()
    #     params = Parameters()
    #     result = wf.run(params=params)
    #     print(result)

    # with slay.run_local():
    #     wf = Workflow(data_splitter=SplitData())
    #     params = Parameters()
    #     result = wf.run(params=params)
    #     print(result)

    # # Gives a `UsageError`, because not in `run_local` context.
    # try:
    #     wf = Workflow()
    # except slay.UsageError:
    #     pass

    # A "marker" to designate which processors should be deployed as public remote
    # service points. Depenedency processors will also be deployed, but only as
    # "internal" services, not as a "public" sevice endpoint.
    slay.deploy_remotely([Workflow])
