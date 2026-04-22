import contextlib
import functools
import json
import sys
from typing import Any, Dict

import requests
import rich
import rich.live
import rich.logging
import rich.spinner
import rich.table
import rich.traceback
from rich.console import Console

from truss.remote.baseten.error import ApiError

rich.spinner.SPINNERS["deploying"] = {"interval": 500, "frames": ["👾 ", " 👾"]}
rich.spinner.SPINNERS["building"] = {"interval": 500, "frames": ["🛠️ ", " 🛠️"]}
rich.spinner.SPINNERS["loading"] = {"interval": 500, "frames": ["⏱️ ", " ⏱️"]}
rich.spinner.SPINNERS["active"] = {"interval": 500, "frames": ["💚 ", " 💚"]}
rich.spinner.SPINNERS["failed"] = {"interval": 500, "frames": ["😤 ", " 😤"]}


console = Console()
error_console = Console(stderr=True, style="bold red")


def json_command(fn):
    """Decorator for CLI commands that support --output json.

    When output_format="json", redirects console to stderr and catches
    exceptions to emit structured JSON errors to stdout.
    In text mode, behaves as a passthrough.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        output_format = kwargs.get("output_format", "text")
        if output_format != "json":
            return fn(*args, **kwargs)

        with console_to_stderr():
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                print(json.dumps(_format_json_error(exc)), file=sys.stdout)
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

    return wrapper


@contextlib.contextmanager
def console_to_stderr():
    """Redirect all Rich consoles to stderr.

    Redirects both the module-level console and Rich's global console
    (used by RichHandler for logging and by progress bars).
    Used in JSON output mode so stdout is reserved for structured JSON.
    """
    global_console = rich.get_console()
    original = console.stderr
    original_global = global_console.stderr
    console.stderr = True
    global_console.stderr = True
    try:
        yield
    finally:
        console.stderr = original
        global_console.stderr = original_global


def _format_json_error(exc: Exception) -> Dict[str, Any]:
    """Format an exception as a JSON-serializable error dict."""
    error: Dict[str, Any] = {"message": str(exc)}

    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        error["status_code"] = exc.response.status_code
        try:
            error["response_body"] = exc.response.json()
        except ValueError:
            error["response_body"] = exc.response.text
    elif isinstance(exc, ApiError):
        error["message"] = exc.message
        if exc.graphql_error_code:
            error["error_code"] = exc.graphql_error_code

    # Include any additional data attached to the exception.
    if hasattr(exc, "json_data") and exc.json_data:
        error.update(exc.json_data)

    return {"error": error}
