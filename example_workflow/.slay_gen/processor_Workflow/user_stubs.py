from slay import stub


class GenerateData(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    def gen_data(self, params: Parameters) -> str:
        json_args = [params.model_dump_json()]
        json_result = self._remote.predict_sync(json_args)
        return json_result


class SplitText(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    async def split(self, data: str, num_partitions: int) -> list[str]:
        json_args = [data, num_partitions]
        json_result = await self._remote.predict_async(json_args)
        return json_result


class TextToNum(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    def to_num(self, data: str, params: Parameters) -> int:
        json_args = [data, params.model_dump_json()]
        json_result = self._remote.predict_sync(json_args)
        return json_result
