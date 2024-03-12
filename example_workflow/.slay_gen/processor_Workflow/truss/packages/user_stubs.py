from slay import stub


class GenerateData(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    def run(self, params: Parameters) -> str:
        json_args = {"params": params.json()}
        json_result = self._remote.predict_sync(json_args)
        return json_result


class SplitText(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    async def run(self, data: str, num_partitions: int) -> list:
        json_args = {"data": data, "num_partitions": num_partitions}
        json_result = await self._remote.predict_async(json_args)
        return json_result


class TextToNum(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    def run(self, data: str, params: Parameters) -> int:
        json_args = {"data": data, "params": params.json()}
        json_result = self._remote.predict_sync(json_args)
        return json_result
