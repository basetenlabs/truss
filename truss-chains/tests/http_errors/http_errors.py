import fastapi

import truss_chains as chains


class FailingHelper(chains.ChainletBase):
    def run_remote(self, max_value: int) -> int:
        raise fastapi.HTTPException(status_code=422, detail="This is a test error.")


@chains.mark_entrypoint
class RaisingEntrypoint(chains.ChainletBase):
    def __init__(self, rand_int=chains.depends(FailingHelper, retries=1)) -> None:
        self._rand_int = rand_int

    def run_remote(self, max_value: int) -> str:
        num_repetitions = self._rand_int.run_remote(max_value)
        return str(num_repetitions)
