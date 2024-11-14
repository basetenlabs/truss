import logging
from typing import AsyncIterator

LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)

logging.basicConfig(
    level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", force=True
)

logging.info("Start")

import asyncio
import time

import pydantic

logging.info("Import Chains")
import truss_chains as chains
from truss_chains import streaming

logging.info("Chains imported")


class Header(pydantic.BaseModel):
    time: float
    msg: str


class MyDataChunk(pydantic.BaseModel):
    words: list[str]
    # numbers: np.ndarray


class Footer(pydantic.BaseModel):
    time: float
    duration_sec: float
    msg: str


class Generator(chains.ChainletBase):
    async def run_remote(self) -> AsyncIterator[bytes]:
        print("Entering Generator")
        streamer = streaming.stream_writer(
            MyDataChunk, header_t=Header, footer_t=Footer
        )
        header = Header(time=time.time(), msg="Start.")
        yield streamer.yield_header(header)
        for i in range(1, 5):
            # numbers = np.full((3, 4), i)
            data = MyDataChunk(
                words=[chr(x + 70) * x for x in range(1, i + 1)],
                # numbers=numbers
            )
            print("Yield")
            # await streamer.yield_header(item)  # TyeError because type mismatch.
            yield streamer.yield_item(data)
            # if i >2:
            #     raise ValueError()
            await asyncio.sleep(0.2)

        end_time = time.time()
        footer = Footer(time=end_time, duration_sec=end_time - header.time, msg="Done.")
        yield streamer.yield_footer(footer)  # TyeError because footer type is None.
        print("Exiting Generator")


class Consumer(chains.ChainletBase):
    def __init__(self, generator=chains.depends(Generator)):
        self._generator = generator

    async def run_remote(self) -> None:
        print("Entering Consumer")
        reader = streaming.StreamReader(
            self._generator.run_remote(), MyDataChunk, header_t=Header, footer_t=Footer
        )
        print("Consuming...")
        header = await reader.read_header()
        print(header)
        async for data in reader.read_items():
            print(f"Read: {data}")

        # reader.yield_item()  # Type error, is reader, not writer.
        # footer = await generator.reader_footer()  # Example does not have a footer.
        print("Exiting Consumer")


logging.info("Module initialized")

if __name__ == "__main__":
    with chains.run_local():
        chain = Consumer()
        result = asyncio.run(chain.run_remote())
        print(result)


    from truss_chains import definitions, remote

    service = remote.push(
        Consumer,
        options=definitions.PushOptionsLocalDocker(
            chain_name="stream", only_generate_trusses=False, use_local_chains_src=True
        ),
    )
    service.run_remote({})
