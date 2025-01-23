import asyncio
import dataclasses
import enum
import struct
import sys
from collections.abc import AsyncIterator
from typing import Generic, Optional, Protocol, Type, TypeVar, overload

import pydantic

_TAG_SIZE = 5  # uint8 + uint32.

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
    item_type: Type[ItemT]
    header_type: HeaderTT  # Is either `Type[HeaderT]` or `None`.
    footer_type: FooterTT  # Is either `Type[FooterT]` or `None`.


@overload
def stream_types(
    item_type: Type[ItemT], *, header_type: Type[HeaderT], footer_type: Type[FooterT]
) -> StreamTypes[ItemT, HeaderT, FooterT]: ...


@overload
def stream_types(
    item_type: Type[ItemT], *, header_type: Type[HeaderT]
) -> StreamTypes[ItemT, HeaderT, None]: ...


@overload
def stream_types(
    item_type: Type[ItemT], *, footer_type: Type[FooterT]
) -> StreamTypes[ItemT, None, FooterT]: ...


@overload
def stream_types(item_type: Type[ItemT]) -> StreamTypes[ItemT, None, None]: ...


def stream_types(
    item_type: Type[ItemT],
    *,
    header_type: Optional[Type[HeaderT]] = None,
    footer_type: Optional[Type[FooterT]] = None,
) -> StreamTypes:
    """Creates a bundle of item type and potentially header/footer types,
    each as pydantic model."""
    # This indirection for creating `StreamTypes` is needed to get generic typing.
    return StreamTypes(item_type, header_type, footer_type)


# Reading ##############################################################################


class _Delimiter(enum.IntEnum):
    NOT_SET = enum.auto()
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
    _footer_data: Optional[bytes]

    async def _read(self) -> tuple[_Delimiter, bytes]: ...


class _StreamReader(_Streamer[ItemT, HeaderTT, FooterTT]):
    _stream: _ByteReader
    _footer_data: Optional[bytes]

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

    async def _read(self) -> tuple[_Delimiter, bytes]:
        try:
            tag = await self._stream.readexactly(_TAG_SIZE)
        # It's ok to read nothing (end of stream), but unexpected to read partial.
        except asyncio.IncompleteReadError:
            raise
        except EOFError:
            return _Delimiter.END, b""

        delimiter, length = self._unpack_tag(tag)
        if not length:
            return delimiter, b""
        data_bytes = await self._stream.readexactly(length)
        return delimiter, data_bytes

    async def read_items(self) -> AsyncIterator[ItemT]:
        delimiter, data_bytes = await self._read()
        if delimiter == _Delimiter.HEADER:
            raise ValueError(
                "Called `read_items`, but there the stream contains header data, which "
                "is not consumed. Call `read_header` first or remove sending a header."
            )
        if delimiter in (_Delimiter.FOOTER, _Delimiter.END):  # In case of 0 items.
            self._footer_data = data_bytes
            return

        assert delimiter == _Delimiter.ITEM
        while True:
            yield self._stream_types.item_type.model_validate_json(data_bytes)
            # We don't know if the next data is another item, footer or the end.
            delimiter, data_bytes = await self._read()
            if delimiter == _Delimiter.END:
                return
            if delimiter == _Delimiter.FOOTER:
                self._footer_data = data_bytes
                return


class _HeaderReadMixin(_Streamer[ItemT, HeaderT, FooterTT]):
    async def read_header(
        self: _StreamReaderProtocol[ItemT, HeaderT, FooterTT],
    ) -> HeaderT:
        delimiter, data_bytes = await self._read()
        if delimiter != _Delimiter.HEADER:
            raise ValueError("Stream does not contain header.")
        return self._stream_types.header_type.model_validate_json(data_bytes)


class _FooterReadMixin(_Streamer[ItemT, HeaderTT, FooterT]):
    _footer_data: Optional[bytes]

    async def read_footer(
        self: _StreamReaderProtocol[ItemT, HeaderTT, FooterT],
    ) -> FooterT:
        if self._footer_data is None:
            delimiter, data_bytes = await self._read()
            if delimiter != _Delimiter.FOOTER:
                raise ValueError("Stream does not contain footer.")
            self._footer_data = data_bytes

        footer = self._stream_types.footer_type.model_validate_json(self._footer_data)
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
    types: StreamTypes[ItemT, None, None], stream: AsyncIterator[bytes]
) -> _StreamReader[ItemT, None, None]: ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, HeaderT, None], stream: AsyncIterator[bytes]
) -> StreamReaderWithHeader[ItemT, HeaderT, None]: ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, None, FooterT], stream: AsyncIterator[bytes]
) -> StreamReaderWithFooter[ItemT, None, FooterT]: ...


@overload
def stream_reader(
    types: StreamTypes[ItemT, HeaderT, FooterT], stream: AsyncIterator[bytes]
) -> StreamReaderFull[ItemT, HeaderT, FooterT]: ...


def stream_reader(
    types: StreamTypes[ItemT, HeaderTT, FooterTT], stream: AsyncIterator[bytes]
) -> _StreamReader:
    if types.header_type is None and types.footer_type is None:
        return _StreamReader(types, stream)
    if types.header_type is None:
        return StreamReaderWithFooter(types, stream)
    if types.footer_type is None:
        return StreamReaderWithHeader(types, stream)

    return StreamReaderFull(types, stream)


# Writing ##############################################################################


class _StreamWriterProtocol(Protocol[ItemT, HeaderTT, FooterTT]):
    _stream_types: StreamTypes[ItemT, HeaderTT, FooterTT]
    _last_sent: _Delimiter

    def _serialize(self, obj: pydantic.BaseModel, delimiter: _Delimiter) -> bytes: ...


class _StreamWriter(_Streamer[ItemT, HeaderTT, FooterTT]):
    def __init__(self, types: StreamTypes[ItemT, HeaderTT, FooterTT]) -> None:
        super().__init__(types)
        self._last_sent = _Delimiter.NOT_SET
        self._stream_types = types

    @staticmethod
    def _pack_tag(delimiter: _Delimiter, length: int) -> bytes:
        return struct.pack(">BI", delimiter.value, length)

    def _serialize(self, obj: pydantic.BaseModel, delimiter: _Delimiter) -> bytes:
        data_bytes = obj.model_dump_json().encode()
        data = bytearray(self._pack_tag(delimiter, len(data_bytes)))
        data.extend(data_bytes)
        # Starlette cannot handle byte array, but view works..
        return memoryview(data)

    def yield_item(self, item: ItemT) -> bytes:
        if self._last_sent in (_Delimiter.FOOTER, _Delimiter.END):
            raise ValueError("Cannot yield item after sending footer / closing stream.")
        self._last_sent = _Delimiter.ITEM
        return self._serialize(item, _Delimiter.ITEM)


class _HeaderWriteMixin(_Streamer[ItemT, HeaderT, FooterTT]):
    def yield_header(
        self: _StreamWriterProtocol[ItemT, HeaderT, FooterTT], header: HeaderT
    ) -> bytes:
        if self._last_sent != _Delimiter.NOT_SET:
            raise ValueError("Cannot yield header after other data has been sent.")
        self._last_sent = _Delimiter.HEADER
        return self._serialize(header, _Delimiter.HEADER)


class _FooterWriteMixin(_Streamer[ItemT, HeaderTT, FooterT]):
    def yield_footer(
        self: _StreamWriterProtocol[ItemT, HeaderTT, FooterT], footer: FooterT
    ) -> bytes:
        if self._last_sent == _Delimiter.END:
            raise ValueError("Cannot yield footer after closing stream.")
        self._last_sent = _Delimiter.FOOTER
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


def stream_writer(types: StreamTypes[ItemT, HeaderTT, FooterTT]) -> _StreamWriter:
    if types.header_type is None and types.footer_type is None:
        return _StreamWriter(types)
    if types.header_type is None:
        return StreamWriterWithFooter(types)
    if types.footer_type is None:
        return StreamWriterWithHeader(types)

    return StreamWriterFull(types)
