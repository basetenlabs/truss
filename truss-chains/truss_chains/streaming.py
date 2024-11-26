import asyncio
import dataclasses
import enum
import struct
import sys
from collections.abc import AsyncIterator
from typing import Generic, Optional, Protocol, Type, TypeVar, overload

import pydantic
from truss.templates.shared import serialization

TAG_SIZE = 5  # uint8 + uint32.
_T = TypeVar("_T")

if sys.version_info < (3, 10):

    async def anext(iterable: AsyncIterator[_T]) -> _T:
        return await iterable.__anext__()


# Note on the (verbose) typing in this module: we want exact typing of the reader and
# writer helpers, while also allowing flexibility to users to leave out header/footer
# if not needed.
# Putting both a constraint on the header/footer types to be pydantic
# models, but also letting them be optional is not well-supported by typing tools,
# (missing feature is using type variables a constraints on other type variables).
#
# A functional, yet verbose workaround that gives correct variadic type inference,
# is using intermediate type variables `HeaderT` <-> `HeaderTT` and in conjunction with
# mapping out all usage combinations with overloads (the overloads essentially allow
# "conditional" binding of type vars). These overloads also allow to use granular
# reader/writer sub-classes conditionally, that have the read/write methods only for the
# data types configured, and implemented DRY with mixin classes.
ItemT = TypeVar("ItemT", bound=pydantic.BaseModel)
HeaderT = TypeVar("HeaderT", bound=pydantic.BaseModel)
FooterT = TypeVar("FooterT", bound=pydantic.BaseModel)

# Since header/footer could also be `None`, we need an extra type variable that
# can assume either `Type[HeaderT]` or `None` - `Type[None]` causes issues.
HeaderTT = TypeVar("HeaderTT")
FooterTT = TypeVar("FooterTT")


@dataclasses.dataclass
class StreamTypes(Generic[ItemT, HeaderTT, FooterTT]):
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


class _Delimiter(enum.IntEnum):
    HEADER = enum.auto()
    ITEM = enum.auto()
    FOOTER = enum.auto()
    END = enum.auto()


class _Streamer(Generic[ItemT, HeaderTT, FooterTT]):
    _stream_types: StreamTypes[ItemT, HeaderTT, FooterTT]

    def __init__(self, types: StreamTypes[ItemT, HeaderTT, FooterTT]) -> None:
        self._stream_types = types


# Reading ##############################################################################


class _ByteReader:
    """Helper to provide `readexactly` API for an async bytes iterator."""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._buffer = bytearray()

    async def readexactly(self, num_bytes: int) -> bytes:
        while len(self._buffer) < num_bytes:
            try:
                chunk = await anext(self._source)
            except StopAsyncIteration:
                break
            self._buffer.extend(chunk)

        if len(self._buffer) < num_bytes:
            if len(self._buffer) == 0:
                raise EOFError()
            raise asyncio.IncompleteReadError(self._buffer, num_bytes)

        result = bytes(self._buffer[:num_bytes])
        del self._buffer[:num_bytes]
        return result


class _StreamReaderProtocol(Protocol[ItemT, HeaderTT, FooterTT]):
    _stream_types: StreamTypes[ItemT, HeaderTT, FooterTT]
    _footer_data: Optional[serialization.MsgPackType]

    async def _read(self) -> tuple[_Delimiter, serialization.MsgPackType]: ...


class _StreamReader(_Streamer[ItemT, HeaderTT, FooterTT]):
    _stream: _ByteReader
    _footer_data: Optional[serialization.MsgPackType]

    def __init__(
        self,
        types: StreamTypes[ItemT, HeaderTT, FooterTT],
        stream: AsyncIterator[bytes],
    ) -> None:
        super().__init__(types)
        self._stream = _ByteReader(stream)
        self._footer_data = None

    @staticmethod
    def _unpack_tag(tag: bytes) -> tuple[_Delimiter, int]:
        enum_value, length = struct.unpack(">BI", tag)
        return _Delimiter(enum_value), length

    async def _read(self) -> tuple[_Delimiter, serialization.MsgPackType]:
        try:
            tag = await self._stream.readexactly(TAG_SIZE)
        # It's ok to read nothing (end of stream), but unexpected to read partial.
        except asyncio.IncompleteReadError:
            raise
        except EOFError:
            return _Delimiter.END, None

        delimiter, length = self._unpack_tag(tag)
        if not length:
            return delimiter, None
        data_bytes = await self._stream.readexactly(length)
        print(f"Read Delimiter: {delimiter}")
        return delimiter, serialization.truss_msgpack_deserialize(data_bytes)

    async def read_items(self) -> AsyncIterator[ItemT]:
        delimiter, data_dict = await self._read()
        if delimiter == _Delimiter.HEADER:
            raise ValueError(
                "Called `read_items`, but there the stream contains header data, which "
                "is not consumed. Call `read_header` first or remove sending a header."
            )
        if delimiter in (_Delimiter.FOOTER, _Delimiter.END):
            return

        assert delimiter == _Delimiter.ITEM
        while True:
            yield self._stream_types.item_t.model_validate(data_dict)
            # We don't know if the next data is another item, footer or the end.
            delimiter, data_dict = await self._read()
            if delimiter == _Delimiter.END:
                return
            if delimiter == _Delimiter.FOOTER:
                self._footer_data = data_dict
                return


class _HeaderReadMixin(_Streamer[ItemT, HeaderT, FooterTT]):
    async def read_header(
        self: _StreamReaderProtocol[ItemT, HeaderT, FooterTT],
    ) -> HeaderT:
        delimiter, data_dict = await self._read()
        if delimiter != _Delimiter.HEADER:
            raise ValueError("Stream does not contain header.")
        return self._stream_types.header_t.model_validate(data_dict)


class _FooterReadMixin(_Streamer[ItemT, HeaderTT, FooterT]):
    _footer_data: Optional[serialization.MsgPackType]

    async def read_footer(
        self: _StreamReaderProtocol[ItemT, HeaderTT, FooterT],
    ) -> FooterT:
        if self._footer_data is None:
            delimiter, data_dict = await self._read()
            if delimiter != _Delimiter.FOOTER:
                raise ValueError("Stream does not contain footer.")
            self._footer_data = data_dict

        footer = self._stream_types.footer_t.model_validate(self._footer_data)
        self._footer_data = None
        return footer


class StreamReaderWithHeader(
    _StreamReader[ItemT, HeaderT, FooterTT], _HeaderReadMixin[ItemT, HeaderT, FooterTT]
): ...


class StreamReaderWithFooter(
    _StreamReader[ItemT, HeaderTT, FooterT], _FooterReadMixin[ItemT, HeaderTT, FooterT]
): ...


class StreamReaderFull(
    _StreamReader[ItemT, HeaderT, FooterT],
    _HeaderReadMixin[ItemT, HeaderT, FooterT],
    _FooterReadMixin[ItemT, HeaderT, FooterT],
): ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, None, None],
    stream: AsyncIterator[bytes],
) -> _StreamReader[ItemT, None, None]: ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, HeaderT, None],
    stream: AsyncIterator[bytes],
) -> StreamReaderWithHeader[ItemT, HeaderT, None]: ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, None, FooterT],
    stream: AsyncIterator[bytes],
) -> StreamReaderWithFooter[ItemT, None, FooterT]: ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, HeaderT, FooterT],
    stream: AsyncIterator[bytes],
) -> StreamReaderFull[ItemT, HeaderT, FooterT]: ...


def stream_reader(
    types: StreamTypes[ItemT, HeaderTT, FooterTT],
    stream: AsyncIterator[bytes],
) -> _StreamReader:
    if types.header_t is None and types.footer_t is None:
        return _StreamReader(types, stream)
    if types.header_t is None:
        return StreamReaderWithFooter(types, stream)
    if types.footer_t is None:
        return StreamReaderWithHeader(types, stream)

    return StreamReaderFull(types, stream)


# Writing ##############################################################################


class _StreamWriterProtocol(Protocol[ItemT, HeaderTT, FooterTT]):
    _stream_types: StreamTypes[ItemT, HeaderTT, FooterTT]

    def _serialize(self, obj: pydantic.BaseModel, delimiter: _Delimiter) -> bytes: ...


class _StreamWriter(_Streamer[ItemT, HeaderTT, FooterTT]):
    @staticmethod
    def _pack_tag(delimiter: _Delimiter, length: int) -> bytes:
        return struct.pack(">BI", delimiter.value, length)

    def _serialize(self, obj: pydantic.BaseModel, delimiter: _Delimiter) -> bytes:
        data_dict = obj.model_dump()
        data_bytes = serialization.truss_msgpack_serialize(data_dict)
        data = bytearray(self._pack_tag(delimiter, len(data_bytes)))
        data.extend(data_bytes)
        # Starlette cannot handle byte array, but view works..
        return memoryview(data)

    def yield_item(self, item: ItemT) -> bytes:
        return self._serialize(item, _Delimiter.ITEM)


class _HeaderWriteMixin(_Streamer[ItemT, HeaderT, FooterTT]):
    def yield_header(
        self: _StreamWriterProtocol[ItemT, HeaderT, FooterTT], header: HeaderT
    ) -> bytes:
        if self._stream_types.header_t is None or header is None:
            raise ValueError()
        return self._serialize(header, _Delimiter.HEADER)


class _FooterWriteMixin(_Streamer[ItemT, HeaderTT, FooterT]):
    def yield_footer(
        self: _StreamWriterProtocol[ItemT, HeaderTT, FooterT], footer: FooterT
    ) -> bytes:
        if self._stream_types.header_t is None or footer is None:
            raise ValueError()
        return self._serialize(footer, _Delimiter.FOOTER)


class StreamWriterWithHeader(
    _StreamWriter[ItemT, HeaderT, FooterTT], _HeaderWriteMixin[ItemT, HeaderT, FooterTT]
): ...


class StreamWriterWithFooter(
    _StreamWriter[ItemT, HeaderTT, FooterT], _FooterWriteMixin[ItemT, HeaderTT, FooterT]
): ...


class StreamWriterFull(
    _StreamWriter[ItemT, HeaderT, FooterT],
    _HeaderWriteMixin[ItemT, HeaderT, FooterT],
    _FooterWriteMixin[ItemT, HeaderT, FooterT],
): ...


@overload
def stream_writer(
    types: StreamTypes[ItemT, None, None],
) -> _StreamWriter[ItemT, None, None]: ...


@overload
def stream_writer(
    types: StreamTypes[ItemT, HeaderT, None],
) -> StreamWriterWithHeader[ItemT, HeaderT, None]: ...


@overload
def stream_writer(
    types: StreamTypes[ItemT, None, FooterT],
) -> StreamWriterWithFooter[ItemT, None, FooterT]: ...


@overload
def stream_writer(
    types: StreamTypes[ItemT, HeaderT, FooterT],
) -> StreamWriterFull[ItemT, HeaderT, FooterT]: ...


def stream_writer(
    types: StreamTypes[ItemT, HeaderTT, FooterTT],
) -> _StreamWriter:
    if types.header_t is None and types.footer_t is None:
        return _StreamWriter(types)
    if types.header_t is None:
        return StreamWriterWithFooter(types)
    if types.footer_t is None:
        return StreamWriterWithHeader(types)

    return StreamWriterFull(types)
