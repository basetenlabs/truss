import asyncio
import logging
import signal
from typing import Callable

from starlette.types import ASGIApp, Receive, Scope, Send

# This is to allow the last request's response to finish handling. There may be more
# middlewares that the response goes through, and then there's the time for the bytes
# to be sent to the caller.
DEFAULT_TERM_DELAY_SECS = 5.0


class TerminationHandlerMiddleware:
    """
    Implements https://www.starlette.io/middleware/#pure-asgi-middleware

    This middleware allows for swiftly and safely terminating the server. It
    listens to a set of termination signals. On receiving such a signal, it terminates
    immediately if there are no outstanding requests and otherwise "marks" the server
    to be terminated when all outstanding requests are done.
    """

    def __init__(
        self,
        app: ASGIApp,
        on_termination: Callable[[], None],
        termination_delay_secs: float = DEFAULT_TERM_DELAY_SECS,
    ):
        self._app = app
        self._outstanding_requests_semaphore = asyncio.Semaphore(0)
        self._on_termination = on_termination
        self._termination_delay_secs = termination_delay_secs
        self._should_terminate_soon = False

        loop = asyncio.get_event_loop()
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            loop.add_signal_handler(sig, self._handle_stop_signal)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            self._outstanding_requests_semaphore.release()  # Increment.
            try:
                await self._app(scope, receive, send)
            finally:
                await self._outstanding_requests_semaphore.acquire()  # Decrement.
                # Check if it's time to terminate after all requests finish
                if (
                    self._should_terminate_soon
                    and self._outstanding_requests_semaphore.locked()
                ):
                    logging.info("Termination after finishing outstanding requests.")
                    # Run in background, to not block the current request handling.
                    asyncio.create_task(self._terminate())
        else:
            await self._app(scope, receive, send)

    def _handle_stop_signal(self) -> None:
        logging.info("Received termination signal.")
        self._should_terminate_soon = True
        if self._outstanding_requests_semaphore.locked():
            logging.info("No outstanding requests. Terminate immediately.")
            asyncio.create_task(self._terminate())
        else:
            logging.info("Will terminate when all requests are processed.")

    async def _terminate(self) -> None:
        logging.info("Sleeping before termination.")
        await asyncio.sleep(self._termination_delay_secs)
        logging.info("Terminating")
        self._on_termination()
