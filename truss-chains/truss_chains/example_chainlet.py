import random

# For more on chains, check out https://docs.baseten.co/chains/overview.
import truss_chains as chains


# By inhereting chains.ChainletBase, the chains framework will know to create a chainlet that hosts the RandInt class.
class RandInt(chains.ChainletBase):
    # run_remote must be implemented by all chainlets. This is the code that will be executed at inference time.
    def run_remote(self, max_value: int) -> int:
        return random.randint(1, max_value)


# The @chains.mark_entrypoint decorator indicates that this Chainlet is the entrypoint.
# Each chain must have exactly one entrypoint.
@chains.mark_entrypoint
class HelloWorld(chains.ChainletBase):
    # chains.depends indicates that the HelloWorld chainlet depends on the RandInt Chainlet
    # this enables the HelloWorld chainlet to call the RandInt chainlet
    def __init__(self, rand_int=chains.depends(RandInt, retries=3)) -> None:
        self._rand_int = rand_int

    def run_remote(self, max_value: int) -> str:
        num_repetitions = self._rand_int.run_remote(max_value)
        return "Hello World! " * num_repetitions


if __name__ == "__main__":
    with chains.run_local():
        hello_world_chain = HelloWorld()
        result = hello_world_chain.run_remote(max_value=5)

    print(result)
    # Hello World! Hello World! Hello World!
