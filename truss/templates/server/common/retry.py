from typing import Callable

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)


def retry(
    fn: Callable,
    count: int,
    logging_fn: Callable,
    base_message: str,
    gap_seconds: float = 0.0,
) -> None:
    attempt = [0]  # mutable pour que le callback voie la valeur à jour

    def _before_sleep(retry_state):
        attempt[0] += 1
        if attempt[0] == 1:
            msg = f"{base_message} Retrying..."
        else:
            msg = f"{base_message} Retrying. Retry count: {attempt[0] - 1}"
        logging_fn(msg)

    r = Retrying(
        stop=stop_after_attempt(count + 1),
        wait=wait_fixed(gap_seconds),
        retry=retry_if_exception_type(Exception),
        before_sleep=_before_sleep if count > 0 else None,
        reraise=True,
    )
    r(fn)