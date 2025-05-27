import anyio
import pytest

anyio.Condition


@pytest.fixture
def anyio_backend():
    return "asyncio"
