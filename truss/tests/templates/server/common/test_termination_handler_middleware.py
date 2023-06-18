import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Awaitable, Callable, List

import pytest
from truss.templates.server.common.termination_handler_middleware import (
    TerminationHandlerMiddleware,
)


async def noop(*args, **kwargs):
    return


@pytest.mark.integration
def test_termination_sequence_no_pending_requests(tmp_path):
    # Create middleware in separate process, on sending term signal to process,
    # it should print the right messages.
    def main_coro_gen(middleware: TerminationHandlerMiddleware):
        import asyncio

        async def main(*args, **kwargs):
            await middleware(1, call_next=noop)
            await asyncio.sleep(1)
            print("should not print due to termination")

        return main()

    _verify_term(main_coro_gen, ["stopped", "terminated"])


@pytest.mark.integration
def test_termination_sequence_with_pending_requests(tmp_path):
    def main_coro_gen(middleware: TerminationHandlerMiddleware):
        import asyncio

        async def main(*args, **kwargs):
            async def call_next(req):
                await asyncio.sleep(1.0)
                return "call_next_called"

            resp = await middleware(1, call_next=call_next)
            print(f"call_next response: {resp}")
            await asyncio.sleep(1)
            print("should not print due to termination")

        return main()

    _verify_term(
        main_coro_gen,
        [
            "stopped",
            "call_next response: call_next_called",
            "terminated",
        ],
    )


def _verify_term(
    main_coro_gen: Callable[[TerminationHandlerMiddleware], Awaitable],
    expected_lines: List[str],
):
    def run(stdout_capture_file_path):
        import asyncio
        import os
        import signal
        import sys

        sys.stdout = open(stdout_capture_file_path, "w")

        def term():
            print("terminated", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)

        middleware = TerminationHandlerMiddleware(
            on_stop=lambda: print("stopped", flush=True),
            on_term=term,
            termination_delay_secs=0.1,
        )
        asyncio.run(main_coro_gen(middleware))

    stdout_capture_file = tempfile.NamedTemporaryFile()
    proc = multiprocessing.Process(target=run, args=(stdout_capture_file.name,))
    proc.start()
    time.sleep(1)
    proc.terminate()
    proc.join(timeout=6.0)
    with Path(stdout_capture_file.name).open() as file:
        lines = [line.strip() for line in file]

    assert lines == expected_lines
