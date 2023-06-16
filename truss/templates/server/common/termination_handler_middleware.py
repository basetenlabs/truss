import signal
import time
from typing import Callable

from fastapi import Request

SELF_KILL_DELAY_SECS = 5


class TerminationHandlerMiddleware:
    def __init__(self, on_stop: Callable[[], None], on_term: Callable[[], None]):
        self._outstanding_request_count = 0
        self._on_stop = on_stop
        self._on_term = on_term
        self._stopeed = False
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            signal.signal(sig, self._stop)

    async def __call__(self, request: Request, call_next):
        self._outstanding_request_count += 1
        try:
            response = await call_next(request)
        finally:
            self._outstanding_request_count -= 1
            if self._outstanding_request_count == 0 and self._stopeed:
                await self._term()
        return response

    def _stop(self, sig, frame):
        self._on_stop()
        self._stopeed = True
        if self._outstanding_request_count == 0:
            self._term()

    def _term(self):
        # Give few seconds for the response flow to finish.
        time.sleep(SELF_KILL_DELAY_SECS)
        self._on_term()
