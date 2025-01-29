import truss_chains as chains


class FirstModel(chains.ModelBase):
    def __init__(self):
        self._call_count = 0

    async def predict(self, call_count_increment: int) -> int:
        self._call_count += call_count_increment
        return self._call_count


class SecondModel(chains.ModelBase):
    def __init__(self):
        self._call_count = 0

    async def predict(self, call_count_increment: int) -> int:
        self._call_count += call_count_increment
        return self._call_count
