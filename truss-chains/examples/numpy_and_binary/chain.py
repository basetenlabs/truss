from typing import Optional

import numpy as np
import pydantic

import truss_chains as chains
from truss_chains import pydantic_numpy


class DataModel(pydantic.BaseModel):
    msg: str
    np_array: pydantic_numpy.NumpyArrayField
    byte_field: Optional[bytes] = None


class SyncChainlet(chains.ChainletBase):
    def run_remote(self, data: DataModel) -> DataModel:
        print(data)
        return data.model_copy(update={"msg": "From sync"})


class AsyncChainlet(chains.ChainletBase):
    async def run_remote(self, data: DataModel) -> DataModel:
        print(data)
        return data.model_copy(update={"msg": "From async"})


class AsyncChainletNoInput(chains.ChainletBase):
    async def run_remote(self, byte_input_internal: bytes) -> DataModel:
        print(byte_input_internal)
        data = DataModel(msg="From async no input", np_array=np.full((2, 2), 3))
        print(data)
        return data


class AsyncChainletNoOutput(chains.ChainletBase):
    async def run_remote(self, data: DataModel) -> None:
        print(data)


class HostJSON(chains.ChainletBase):
    """Calls various chainlets in JSON mode."""

    def __init__(
        self,
        sync_chainlet=chains.depends(SyncChainlet, use_binary=False),
        async_chainlet=chains.depends(AsyncChainlet, use_binary=False),
        async_chainlet_no_output=chains.depends(
            AsyncChainletNoOutput, use_binary=False
        ),
        async_chainlet_no_input=chains.depends(AsyncChainletNoInput, use_binary=False),
    ):
        self._sync_chainlet = sync_chainlet
        self._async_chainlet = async_chainlet
        self._async_chainlet_no_output = async_chainlet_no_output
        self._async_chainlet_no_input = async_chainlet_no_input

    async def run_remote(self) -> tuple[DataModel, DataModel, DataModel]:
        a = np.ones((3, 2, 1))
        data = DataModel(msg="From Host", np_array=a)
        sync_result = self._sync_chainlet.run_remote(data)
        print(sync_result)
        async_result = await self._async_chainlet.run_remote(data)
        print(async_result)
        await self._async_chainlet_no_output.run_remote(data)
        async_no_input = await self._async_chainlet_no_input.run_remote()
        print(async_no_input)
        return sync_result, async_result, async_no_input


class HostBinary(chains.ChainletBase):
    """Calls various chainlets in binary mode."""

    def __init__(
        self,
        sync_chainlet=chains.depends(SyncChainlet, use_binary=True),
        async_chainlet=chains.depends(AsyncChainlet, use_binary=True),
        async_chainlet_no_output=chains.depends(AsyncChainletNoOutput, use_binary=True),
        async_chainlet_no_input=chains.depends(AsyncChainletNoInput, use_binary=True),
    ):
        self._sync_chainlet = sync_chainlet
        self._async_chainlet = async_chainlet
        self._async_chainlet_no_output = async_chainlet_no_output
        self._async_chainlet_no_input = async_chainlet_no_input

    async def run_remote(
        self, byte_input: bytes
    ) -> tuple[DataModel, DataModel, DataModel]:
        print(byte_input)
        a = np.ones((3, 2, 1))
        data = DataModel(msg="From Host", np_array=a)
        sync_result = self._sync_chainlet.run_remote(data)
        print(sync_result)
        async_result = await self._async_chainlet.run_remote(data)
        print(async_result)
        await self._async_chainlet_no_output.run_remote(data)
        async_no_input = await self._async_chainlet_no_input.run_remote(byte_input)
        print(async_no_input)
        return sync_result, async_result, async_no_input
