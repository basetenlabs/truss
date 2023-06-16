import asyncio
import os
import signal
from typing import Optional

from fastapi import Request

SELF_KILL_DELAY_SECS = 5


class TerminationHandlerMiddleware:
    def __init__(self, server):
        self._outstanding_request_count = 0
        self._server = server
        self._stopeed = False
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self._stop(sig=s))
            )

    async def __call__(self, request: Request, call_next):
        self._outstanding_request_count += 1
        try:
            response = await call_next(request)
        finally:
            self._outstanding_request_count -= 1
            if self._outstanding_request_count == 0 and self._stopeed:
                self._kill_self()
        return response

    def _stop(self, sig: Optional[int] = None):
        self._server.stop()
        self._stopeed = True

    async def _kill_self(self):
        # Give few seconds for the response to exit.
        await asyncio.sleep(SELF_KILL_DELAY_SECS)
        os.kill(os.getpid(), signal.SIGKILL)
