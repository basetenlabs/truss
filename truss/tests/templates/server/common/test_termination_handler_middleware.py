import multiprocessing
import tempfile
import time
from pathlib import Path

from truss.templates.server.common.termination_handler_middleware import (
    TerminationHandlerMiddleware,
)


async def noop(*args, **kwargs):
    return


def test_termination_sequence():
    # Create middleware in separate process, on sending term signal to process,
    # it should print the right messages.
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
        )

        async def main(*args, **kwargs):
            await middleware(1, call_next=noop)
            await middleware(2, call_next=noop)
            for _ in range(100):
                await asyncio.sleep(1)

        asyncio.run(main())

    stdout_capture_file = tempfile.NamedTemporaryFile()
    proc = multiprocessing.Process(target=run, args=(stdout_capture_file.name,))
    proc.start()
    time.sleep(1)
    proc.terminate()
    proc.join(timeout=6.0)
    with Path(stdout_capture_file.name).open() as file:
        lines = [line.strip() for line in file]

    print(lines)
    assert lines[0] == "stopped"
    assert lines[1] == "terminated"
