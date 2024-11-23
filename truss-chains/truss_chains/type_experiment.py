from typing import Generic, NamedTuple, Optional, Type, TypeVar, overload

import pydantic
from typing_extensions import reveal_type

ItemT = TypeVar("ItemT", bound=pydantic.BaseModel)
HeaderT = TypeVar("HeaderT")


class StreamTypes(NamedTuple, Generic[ItemT, HeaderT]):
    item_t: Type[ItemT]
    header_t: HeaderT


@overload
def stream_types(item_t: Type[ItemT]) -> StreamTypes[ItemT, None]: ...


@overload
def stream_types(
    item_t: Type[ItemT], *, header_t: Type[HeaderT]
) -> StreamTypes[ItemT, Type[HeaderT]]: ...


def stream_types(item_t: Type[ItemT], *, header_t: Optional[Type[HeaderT]] = None):
    return StreamTypes(item_t, header_t)


class _Streamer(Generic[ItemT, HeaderT]):
    _stream_types: StreamTypes[ItemT, HeaderT]

    def __init__(self, stream_types_: StreamTypes[ItemT, HeaderT]) -> None:
        self._stream_types = stream_types_


if __name__ == "__main__":

    class Header(pydantic.BaseModel):
        time: float
        msg: str

    class MyDataChunk(pydantic.BaseModel):
        words: list[str]

    NONE_TYPES = stream_types(MyDataChunk)
    FULL_TYPES = stream_types(MyDataChunk, header_t=Header)

    streamer_none = _Streamer(NONE_TYPES)
    reveal_type(streamer_none._stream_types.item_t)
    reveal_type(streamer_none._stream_types.header_t)  # Revealed type is 'None'

    streamer_full = _Streamer(FULL_TYPES)
    reveal_type(streamer_full._stream_types.header_t)  # Revealed type is 'Type[Header]'
