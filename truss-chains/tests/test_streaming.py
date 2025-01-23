import asyncio
from typing import AsyncIterator

import pydantic
import pytest

from truss_chains import streaming


class Header(pydantic.BaseModel):
    time: float
    msg: str


class MyDataChunk(pydantic.BaseModel):
    words: list[str]


class Footer(pydantic.BaseModel):
    time: float
    duration_sec: float
    msg: str


async def to_bytes_iterator(data_stream) -> AsyncIterator[bytes]:
    for data in data_stream:
        yield data
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_streaming_with_header_and_footer():
    types = streaming.stream_types(
        item_type=MyDataChunk, header_type=Header, footer_type=Footer
    )

    writer = streaming.stream_writer(types)
    header = Header(time=123.456, msg="Start of stream")
    items = [
        MyDataChunk(words=["hello", "world"]),
        MyDataChunk(words=["foo", "bar"]),
        MyDataChunk(words=["baz"]),
    ]
    footer = Footer(time=789.012, duration_sec=665.556, msg="End of stream")

    data_stream = []
    data_stream.append(writer.yield_header(header))
    for item in items:
        data_stream.append(writer.yield_item(item))
    data_stream.append(writer.yield_footer(footer))

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    # Assert that serialization roundtrip works.
    read_header = await reader.read_header()
    assert read_header == header
    read_items = []
    async for item in reader.read_items():
        read_items.append(item)
    assert read_items == items
    read_footer = await reader.read_footer()
    assert read_footer == footer


@pytest.mark.asyncio
async def test_streaming_with_items_only():
    types = streaming.stream_types(item_type=MyDataChunk)
    writer = streaming.stream_writer(types)

    items = [
        MyDataChunk(words=["hello", "world"]),
        MyDataChunk(words=["foo", "bar"]),
        MyDataChunk(words=["baz"]),
    ]

    data_stream = []
    for item in items:
        data_stream.append(writer.yield_item(item))

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    read_items = []
    async for item in reader.read_items():
        read_items.append(item)

    assert read_items == items


@pytest.mark.asyncio
async def test_reading_header_when_none_sent():
    types = streaming.stream_types(item_type=MyDataChunk, header_type=Header)
    writer = streaming.stream_writer(types)
    items = [MyDataChunk(words=["hello", "world"])]

    data_stream = []
    for item in items:
        data_stream.append(writer.yield_item(item))

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    with pytest.raises(ValueError, match="Stream does not contain header."):
        await reader.read_header()


@pytest.mark.asyncio
async def test_reading_items_with_wrong_model():
    types_writer = streaming.stream_types(item_type=MyDataChunk)
    types_reader = streaming.stream_types(item_type=Header)  # Wrong item type
    writer = streaming.stream_writer(types_writer)
    items = [MyDataChunk(words=["hello", "world"])]
    data_stream = []
    for item in items:
        data_stream.append(writer.yield_item(item))

    reader = streaming.stream_reader(types_reader, to_bytes_iterator(data_stream))

    with pytest.raises(pydantic.ValidationError):
        async for item in reader.read_items():
            pass


@pytest.mark.asyncio
async def test_streaming_with_wrong_order():
    types = streaming.stream_types(
        item_type=MyDataChunk, header_type=Header, footer_type=Footer
    )

    writer = streaming.stream_writer(types)
    header = Header(time=123.456, msg="Start of stream")
    items = [MyDataChunk(words=["hello", "world"])]
    footer = Footer(time=789.012, duration_sec=665.556, msg="End of stream")
    data_stream = []
    for item in items:
        data_stream.append(writer.yield_item(item))

    with pytest.raises(
        ValueError, match="Cannot yield header after other data has been sent."
    ):
        data_stream.append(writer.yield_header(header))
    data_stream.append(writer.yield_footer(footer))

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    # Try to read header, should fail because the first data is an item
    with pytest.raises(ValueError, match="Stream does not contain header."):
        await reader.read_header()


@pytest.mark.asyncio
async def test_reading_items_without_consuming_header():
    types = streaming.stream_types(item_type=MyDataChunk, header_type=Header)
    writer = streaming.stream_writer(types)
    header = Header(time=123.456, msg="Start of stream")
    items = [MyDataChunk(words=["hello", "world"])]

    data_stream = []
    data_stream.append(writer.yield_header(header))
    for item in items:
        data_stream.append(writer.yield_item(item))

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    # Try to read items without consuming header
    with pytest.raises(
        ValueError,
        match="Called `read_items`, but there the stream contains header data",
    ):
        async for item in reader.read_items():
            pass


@pytest.mark.asyncio
async def test_reading_footer_when_none_sent():
    types = streaming.stream_types(item_type=MyDataChunk, footer_type=Footer)
    writer = streaming.stream_writer(types)
    items = [MyDataChunk(words=["hello", "world"])]
    data_stream = []
    for item in items:
        data_stream.append(writer.yield_item(item))

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    read_items = []
    async for item in reader.read_items():
        read_items.append(item)
    assert read_items == items

    # Try to read footer, expect an error
    with pytest.raises(ValueError, match="Stream does not contain footer."):
        await reader.read_footer()


@pytest.mark.asyncio
async def test_reading_footer_with_no_items():
    types = streaming.stream_types(item_type=MyDataChunk, footer_type=Footer)
    writer = streaming.stream_writer(types)
    footer = Footer(time=789.012, duration_sec=665.556, msg="End of stream")
    data_stream = [writer.yield_footer(footer)]

    reader = streaming.stream_reader(types, to_bytes_iterator(data_stream))
    read_items = []
    async for item in reader.read_items():
        read_items.append(item)
    assert len(read_items) == 0

    read_footer = await reader.read_footer()
    assert read_footer == footer
