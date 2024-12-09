import asyncio
import time
from typing import AsyncIterator

import pydantic

import truss_chains as chains
from truss_chains import streaming


class Header(pydantic.BaseModel):
    time: float
    msg: str


class MyDataChunk(pydantic.BaseModel):
    words: list[str]


class Footer(pydantic.BaseModel):
    time: float
    duration_sec: float
    msg: str


class ConsumerOutput(pydantic.BaseModel):
    header: Header
    chunks: list[MyDataChunk]
    footer: Footer
    strings: str


STREAM_TYPES = streaming.stream_types(
    MyDataChunk, header_type=Header, footer_type=Footer
)


class Generator(chains.ChainletBase):
    """Example that streams fully structured pydantic items with header and footer."""

    async def run_remote(self, cause_error: bool) -> AsyncIterator[bytes]:
        print("Entering Generator")
        streamer = streaming.stream_writer(STREAM_TYPES)
        header = Header(time=time.time(), msg="Start.")
        yield streamer.yield_header(header)
        for i in range(1, 5):
            data = MyDataChunk(
                words=[chr(x + 70) * x for x in range(1, i + 1)],
            )
            print("Yield")
            yield streamer.yield_item(data)
            if cause_error and i > 2:
                raise RuntimeError("Test Error")
            await asyncio.sleep(0.05)

        end_time = time.time()
        footer = Footer(time=end_time, duration_sec=end_time - header.time, msg="Done.")
        yield streamer.yield_footer(footer)
        print("Exiting Generator")


class StringGenerator(chains.ChainletBase):
    """Minimal streaming example with strings (e.g. for raw LLM output)."""

    async def run_remote(self) -> AsyncIterator[str]:
        # Note: the "chunk" boundaries are lost, when streaming raw strings. You must
        # add spaces and linebreaks to the items yourself..
        yield "First "
        yield "second "
        yield "last."


class Consumer(chains.ChainletBase):
    """Consume that reads the raw streams and parses them."""

    def __init__(
        self,
        generator=chains.depends(Generator),
        string_generator=chains.depends(StringGenerator),
    ):
        self._generator = generator
        self._string_generator = string_generator

    async def run_remote(self, cause_error: bool) -> ConsumerOutput:
        print("Entering Consumer")
        reader = streaming.stream_reader(
            STREAM_TYPES, self._generator.run_remote(cause_error)
        )
        print("Consuming...")
        header = await reader.read_header()
        chunks = []
        async for data in reader.read_items():
            print(f"Read: {data}")
            chunks.append(data)

        footer = await reader.read_footer()
        strings = []
        async for part in self._string_generator.run_remote():
            strings.append(part)

        print("Exiting Consumer")
        return ConsumerOutput(
            header=header, chunks=chunks, footer=footer, strings="".join(strings)
        )


if __name__ == "__main__":
    with chains.run_local():
        chain = Consumer()
        result = asyncio.run(chain.run_remote(False))
        print(result)
