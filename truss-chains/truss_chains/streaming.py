import enum
import struct
import sys
from collections.abc import AsyncIterator
from typing import NamedTuple, Optional, Protocol, overload

import pydantic
from truss.templates.shared import serialization
from typing_extensions import Generic, Type, TypeVar

TAG_SIZE = 5  # uint8 + uint32.
_T = TypeVar("_T")

if sys.version_info < (3, 10):

    async def anext(iterable: AsyncIterator[_T]) -> _T:
        return await iterable.__anext__()


ItemT = TypeVar("ItemT", bound=pydantic.BaseModel)
HeaderT = TypeVar("HeaderT", bound=pydantic.BaseModel)
FooterT = TypeVar("FooterT", bound=pydantic.BaseModel)

# Since header/footer could also be None, we need an extra type variable that
# can assume either `Type[HeaderT]` or `None` - `Type[None]` would not work.
HeaderTT = TypeVar("HeaderTT")
FooterTT = TypeVar("FooterTT")


class StreamTypes(NamedTuple, Generic[ItemT, HeaderTT, FooterTT]):
    item_t: Type[ItemT]
    header_t: HeaderTT  # Is either `Type[HeaderT]` or `None`.
    footer_t: FooterTT  # Is either `Type[FooterT]` or `None`.


@overload
def stream_types(
    item_t: Type[ItemT],
    *,
    header_t: Type[HeaderT],
    footer_t: Type[FooterT],
) -> StreamTypes[ItemT, HeaderT, FooterT]: ...


@overload
def stream_types(
    item_t: Type[ItemT],
    *,
    header_t: Type[HeaderT],
) -> StreamTypes[ItemT, HeaderT, None]: ...


@overload
def stream_types(
    item_t: Type[ItemT],
    *,
    footer_t: Type[FooterT],
) -> StreamTypes[ItemT, None, FooterT]: ...


@overload
def stream_types(item_t: Type[ItemT]) -> StreamTypes[ItemT, None, None]: ...


def stream_types(
    item_t: Type[ItemT],
    *,
    header_t: Optional[Type[HeaderT]] = None,
    footer_t: Optional[Type[FooterT]] = None,
) -> StreamTypes:
    """Creates a bundle of item type and potentially header/footer types,
    each as pydantic model."""
    # This indirection for creating `StreamTypes` is needed to get generic typing.
    return StreamTypes(item_t, header_t, footer_t)


# Reading ##############################################################################


class Delimiter(enum.IntEnum):
    HEADER = enum.auto()
    ITEM = enum.auto()
    FOOTER = enum.auto()
    END = enum.auto()


class _Streamer(Generic[ItemT, HeaderT, FooterT]):
    _stream_types: StreamTypes[ItemT, HeaderT, FooterT]

    def __init__(self, stream_types: StreamTypes[ItemT, HeaderT, FooterT]) -> None:
        self._stream_types = stream_types


# Reading ##############################################################################


class _ByteReader:
    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._buffer = bytearray()

    async def readexactly(self, num_bytes: int) -> bytes:
        while len(self._buffer) < num_bytes:
            try:
                chunk = await anext(self._source)
            except StopAsyncIteration:
                if len(self._buffer) < num_bytes:
                    raise EOFError(
                        f"Requested to read `{num_bytes}` bytes, "
                        f"but only `{len(self._buffer)}` available"
                    )
                break
            self._buffer.extend(chunk)

        result = bytes(self._buffer[:num_bytes])
        del self._buffer[:num_bytes]
        return result


class _StreamReaderProtocol(Protocol[ItemT, HeaderT, FooterT]):
    async def _read(self) -> tuple[Delimiter, serialization.MsgPackType]: ...

    _footer_data: Optional[serialization.MsgPackType]
    _stream_types: StreamTypes[ItemT, HeaderT, FooterT]


class StreamReader(_Streamer[ItemT, HeaderT, FooterT]):
    _stream: _ByteReader
    _footer_data: Optional[serialization.MsgPackType]

    def __init__(
        self,
        stream_types: StreamTypes[ItemT, HeaderT, FooterT],
        stream: AsyncIterator[bytes],
    ) -> None:
        super().__init__(stream_types)
        self._stream = _ByteReader(stream)
        self._footer_data = None

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

    async def read_items(self) -> AsyncIterator[ItemT]:
        delimiter, data_dict = await self._read()
        assert delimiter == Delimiter.ITEM

        while delimiter == Delimiter.ITEM:
            yield self._stream_types.item_t.model_validate(data_dict)
            # Read next: either item, footer, or end.
            delimiter, data_dict = await self._read()
            if delimiter == Delimiter.END:
                return
            if delimiter == Delimiter.FOOTER:
                self._footer_data = data_dict
                return


class _HeaderReadMixin(_Streamer[ItemT, HeaderT, FooterT]):
    async def read_header(
        self: _StreamReaderProtocol[ItemT, HeaderT, FooterT],
    ) -> HeaderT:
        delimiter, data_dict = await self._read()
        assert delimiter == Delimiter.HEADER
        return self._stream_types.header_t.model_validate(data_dict)


class _FooterReadMixin(_Streamer[ItemT, HeaderT, FooterT]):
    _footer_data: Optional[serialization.MsgPackType]

    async def read_footer(
        self: _StreamReaderProtocol[ItemT, HeaderT, FooterT],
    ) -> FooterT:
        if self._footer_data is None:
            raise ValueError()
        footer = self._stream_types.footer_t.model_validate(self._footer_data)
        self._footer_data = None
        return footer


class StreamReaderWithHeader(
    StreamReader[ItemT, HeaderT, FooterT], _HeaderReadMixin[ItemT, HeaderT, FooterT]
): ...


class StreamReaderWithFooter(
    StreamReader[ItemT, HeaderT, FooterT], _HeaderReadMixin[ItemT, HeaderT, FooterT]
): ...


class StreamReaderFull(
    StreamReader[ItemT, HeaderT, FooterT],
    _HeaderReadMixin[ItemT, HeaderT, FooterT],
    _FooterReadMixin[ItemT, HeaderT, FooterT],
): ...


@overload
def stream_reader(
    stream_types: StreamTypes[ItemT, None, None],
    stream: AsyncIterator[bytes],
) -> StreamReader[ItemT, None, None]: ...


@overload
def stream_reader(
    stream_types: StreamTypes[ItemT, HeaderT, None],
    stream: AsyncIterator[bytes],
) -> StreamReaderWithHeader[ItemT, HeaderT, None]: ...


@overload
def stream_reader(
    stream_types: StreamTypes[ItemT, None, FooterT],
    stream: AsyncIterator[bytes],
) -> StreamReaderWithFooter[ItemT, None, FooterT]: ...


@overload
def stream_reader(
    stream_types: StreamTypes[ItemT, HeaderT, FooterT],
    stream: AsyncIterator[bytes],
) -> StreamReaderFull[ItemT, HeaderT, FooterT]: ...


def stream_reader(
    stream_types: StreamTypes[ItemT, HeaderT, FooterT],
    stream: AsyncIterator[bytes],
) -> StreamReader:
    if stream_types.header_t is None and stream_types.footer_t is None:
        return StreamReader(stream_types, stream)
    if stream_types.header_t is None:
        return StreamReaderWithFooter(stream_types, stream)
    if stream_types.footer_t is None:
        return StreamReaderWithHeader(stream_types, stream)
    return StreamReaderFull(stream_types, stream)


# Writing ##############################################################################


class StreamWriter(_Streamer[ItemT, HeaderT, FooterT]):
    @staticmethod
    def _pack_tag(delimiter: Delimiter, length: int) -> bytes:
        return struct.pack(">BI", delimiter.value, length)

    def _serialize(self, obj: pydantic.BaseModel, delimiter: Delimiter) -> bytes:
        data_dict = obj.model_dump()
        data_bytes = serialization.truss_msgpack_serialize(data_dict)
        data = bytearray(self._pack_tag(delimiter, len(data_bytes)))
        data.extend(data_bytes)
        print(data)
        # Starlette cannot handle byte array.
        return memoryview(data)

    def yield_header(self, header: HeaderT) -> bytes:
        if self._stream_types.header_t is None or header is None:
            raise ValueError()
        return self._serialize(header, Delimiter.HEADER)

    def yield_item(self, item: ItemT) -> bytes:
        return self._serialize(item, Delimiter.ITEM)

    def yield_footer(self, footer: FooterT) -> bytes:
        if self._stream_types.header_t is None or footer is None:
            raise ValueError()
        return self._serialize(footer, Delimiter.FOOTER)
