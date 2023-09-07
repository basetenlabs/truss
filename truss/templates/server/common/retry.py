import time
from typing import Callable, Optional


def retry(
    fn: Callable,
    count: int,
    logging_fn: Callable,
    base_message: str,
    gap_seconds: float = 0.0,
    logging_exc_fn: Optional[Callable] = None,
):
    i = 0
    while i <= count:
        try:
            fn()
            return
        except Exception as exc:
            if logging_exc_fn:
                logging_exc_fn(exc)
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
