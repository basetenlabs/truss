import asyncio
import time
from typing import Callable


# TODO: replace with tenacity.
def retry(
    fn: Callable,
    count: int,
    logging_fn: Callable,
    base_message: str,
    gap_seconds: float = 0.0,
):
    i = 0
    while i <= count:
        try:
            fn()
            return
        except Exception as exc:
            msg = base_message
            if i >= count:
                raise exc

            if i == 0:
                msg = f"{msg} Retrying..."
            else:
                msg = f"{msg} Retrying. Retry count: {i}"
            logging_fn(msg)
            i += 1
            time.sleep(gap_seconds)


async def retry_async(
    fn: Callable,
    count: int,
    logging_fn: Callable,
    base_message: str,
    gap_seconds: float = 0.0,
):
    i = 0
    while i <= count:
        try:
            await fn()
            return
        except Exception as exc:
            msg = base_message
            if i >= count:
                raise exc

            if i == 0:
                msg = f"{msg} Retrying..."
            else:
                msg = f"{msg} Retrying. Retry count: {i}"
            logging_fn(msg)
            i += 1
            await asyncio.sleep(gap_seconds)
