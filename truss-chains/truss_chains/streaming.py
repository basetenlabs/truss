import asyncio
import enum
import struct
import sys
from collections.abc import AsyncIterator
from typing import Optional, Protocol, overload

import pydantic
from truss.templates.shared import serialization
from typing_extensions import Generic, Type, TypeVar, runtime_checkable

_T = TypeVar("_T")

if sys.version_info < (3, 10):

    async def anext(iterable: AsyncIterator[_T]) -> _T:
        return await iterable.__anext__()


# Type variables for Header, Item, and Footer
TItem = TypeVar("TItem", bound=pydantic.BaseModel)
THeader = TypeVar("THeader", pydantic.BaseModel, None)
TFooter = TypeVar("TFooter", pydantic.BaseModel, None)

TAG_SIZE = 5  # uint8 + uint32.


@runtime_checkable
class _StreamReaderLike(Protocol):
    async def readexactly(self, num_bytes: int) -> bytes: ...


class _ByteReader:
    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._buffer = bytearray()
        self._lock = asyncio.Lock()

    async def readexactly(self, num_bytes: int) -> bytes:
        async with self._lock:
            while len(self._buffer) < num_bytes:
                try:
                    chunk = await anext(self._source)
                except StopAsyncIteration:
                    break
                self._buffer.extend(chunk)

            if len(self._buffer) < num_bytes:
                raise EOFError("TODO")

            result = bytes(self._buffer[:num_bytes])
            del self._buffer[:num_bytes]
            return result


class Delimiter(enum.IntEnum):
    HEADER = enum.auto()
    ITEM = enum.auto()
    FOOTER = enum.auto()
    END = enum.auto()


class Streamer(Generic[TItem, THeader, TFooter]):
    _item_t: Type[TItem]
    _header_t: Optional[Type[THeader]]
    _footer_t: Optional[Type[TFooter]]

    def __init__(
        self,
        item_t: Type[TItem],
        header_t: Optional[Type[THeader]],
        footer_t: Optional[Type[TFooter]],
    ) -> None:
        self._item_t = item_t
        self._header_t = header_t
        self._footer_t = footer_t


class StreamReader(Streamer[TItem, THeader, TFooter]):
    _stream: _StreamReaderLike

    def __init__(
        self,
        stream: AsyncIterator[bytes],
        item_t: Type[TItem],
        header_t: Optional[Type[THeader]],
        footer_t: Optional[Type[TFooter]],
    ) -> None:
        super().__init__(item_t, header_t, footer_t)
        self._stream = _ByteReader(stream)
        self._footer = None

    @staticmethod
    def _unpack_tag(tag: bytes) -> tuple[Delimiter, int]:
        enum_value, length = struct.unpack(">BI", tag)
        return Delimiter(enum_value), length

    async def _read(self) -> tuple[Delimiter, serialization.MsgPackType]:
        tag = await self._stream.readexactly(TAG_SIZE)
        delimiter, length = self._unpack_tag(tag)
        if not length:
            return delimiter, None
        data_bytes = await self._stream.readexactly(length)
        return delimiter, serialization.truss_msgpack_deserialize(data_bytes)

    async def read_header(self) -> THeader:
        delimiter, data_dict = await self._read()
        assert delimiter == Delimiter.HEADER
        return self._header_t.parse_obj(data_dict)

    async def read_items(self) -> AsyncIterator[TItem]:
        delimiter, data_dict = await self._read()
        assert delimiter == Delimiter.ITEM

        while delimiter == Delimiter.ITEM:
            yield self._item_t.parse_obj(data_dict)
            # Read next: either item, footer, or end.
            delimiter, data_dict = await self._read()
            if delimiter == Delimiter.END:
                return
            if delimiter == Delimiter.FOOTER:
                self._footer = self._footer_t.parse_obj(data_dict)
                return

    async def read_footer(self) -> TFooter:
        if self._footer_t is None:
            raise ValueError()
        footer = self._footer_t
        self._footer_t = None
        return footer


class StreamWriter(Streamer[TItem, THeader, TFooter]):
    @staticmethod
    def _pack_tag(delimiter: Delimiter, length: int) -> bytes:
        return struct.pack(">BI", delimiter.value, length)

    def _serialize(self, obj: pydantic.BaseModel, delimiter: Delimiter) -> bytes:
        data_dict = obj.dict()
        data_bytes = serialization.truss_msgpack_serialize(data_dict)
        data = bytearray(self._pack_tag(delimiter, len(data_bytes)))
        data.extend(data_bytes)
        print(data)
        # Starlette cannot handle byte array.
        return memoryview(data)

    def yield_header(self, header: THeader) -> bytes:
        return self._serialize(header, Delimiter.HEADER)

    def yield_item(self, item: TItem) -> bytes:
        return self._serialize(item, Delimiter.ITEM)

    def yield_footer(self, footer: TFooter) -> bytes:
        return self._serialize(footer, Delimiter.FOOTER)


@overload
def stream_writer(
    item_t: Type[TItem],
    *,
    header_t: Type[THeader],
    footer_t: Type[TFooter],
) -> StreamWriter[TItem, THeader, TFooter]: ...


@overload
def stream_writer(
    item_t: Type[TItem],
    *,
    header_t: Type[THeader],
) -> StreamWriter[TItem, THeader, None]: ...


@overload
def stream_writer(
    item_t: Type[TItem],
    *,
    footer_t: Type[TFooter],
) -> StreamWriter[TItem, None, TFooter]: ...


@overload
def stream_writer(item_t: Type[TItem]) -> StreamWriter[TItem, None, None]: ...


def stream_writer(
    item_t: Type[TItem],
    *,
    header_t: Optional[Type[THeader]] = None,
    footer_t: Optional[Type[TFooter]] = None,
) -> StreamWriter:
    return StreamWriter(item_t, header_t, footer_t)
