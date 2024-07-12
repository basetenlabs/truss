import asyncio
import random
import truss_chains as chains


# This Chainlet does the work
class SayHello(chains.ChainletBase):

    async def run_remote(self, name: str) -> str:
        # random sleep between 0 and 2 seconds
        with self.tracer.start_as_current_span("doing_work"):
            await asyncio.sleep(random.random() * 1)
        with self.tracer.start_as_current_span("end_work"):
            await asyncio.sleep(random.random() * 1)
        return f"Hello, {name}"


# This Chainlet orchestrates the work
@chains.mark_entrypoint
class HelloAll(chains.ChainletBase):

    def __init__(self, say_hello_chainlet=chains.depends(SayHello)) -> None:
        self._say_hello = say_hello_chainlet

    async def run_remote(self, names: list[str]) -> str:
        tasks = []
        with self.tracer.start_as_current_span("preprocessing"):
            await asyncio.sleep(1)

        for name in names:
            tasks.append(asyncio.ensure_future(
                self._say_hello.run_remote(name)))
        
        result = "\n".join(await asyncio.gather(*tasks))
        with self.tracer.start_as_current_span("postprocessing"):
            await asyncio.sleep(1)
        return result


# Test the Chain locally
if __name__ == "__main__":
    with chains.run_local():
        hello_chain = HelloAll()
        result = asyncio.get_event_loop().run_until_complete(
            hello_chain.run_remote(["Marius", "Sid", "Bola"]))
        print(result)
