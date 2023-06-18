import asyncio
import signal
from typing import Callable

from fastapi import Request

DEFAULT_TERM_DELAY_SECS = 5.0


class TerminationHandlerMiddleware:
    def __init__(
        self,
        on_stop: Callable[[], None],
        on_term: Callable[[], None],
        termination_delay_secs: float = DEFAULT_TERM_DELAY_SECS,
    ):
        self._outstanding_request_count = 0
        self._on_stop = on_stop
        self._on_term = on_term
        self._termination_delay_secs = termination_delay_secs
        self._stopped = False
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            signal.signal(sig, self._stop)

    async def __call__(self, request: Request, call_next):
        self._outstanding_request_count += 1
        try:
            response = await call_next(request)
        finally:
            self._outstanding_request_count -= 1
            if self._outstanding_request_count == 0 and self._stopped:
                # There's a delay in term to allow some time for current
                # response flow to finish.
                asyncio.create_task(self._term())
        return response

    def _stop(self, sig, frame):
        self._on_stop()
        self._stopped = True
        if self._outstanding_request_count == 0:
            self._on_term()

    async def _term(self):
        await asyncio.sleep(self._termination_delay_secs)
        self._on_term()
