import datetime
import logging
import os
import re
import sys
import threading
import time
import warnings
from functools import wraps
from typing import Any, Callable, Optional

import pydantic
import requests as requests_lib
import rich
import rich.live
import rich.logging
import rich.spinner
import rich.table
import rich.traceback
import rich_click as click
from rich.markup import escape

import truss
from truss.cli.cli import rich_console
from truss.cli.utils import self_upgrade
from truss.cli.utils.output import console
from truss.remote.baseten.core import ACTIVE_STATUS, DEPLOYING_STATUSES
from truss.remote.baseten.remote import BasetenRemote

logger = logging.getLogger(__name__)

INCLUDE_GIT_INFO_DOC = (
    "Whether to attach git versioning info (sha, branch, tag) to deployments made from "
    "within a git repo. If set to True in `.trussrc`, it will always be attached."
)

_HUMANFRIENDLY_LOG_LEVEL = "humanfriendly"
_log_level_str_to_level = {
    _HUMANFRIENDLY_LOG_LEVEL: logging.INFO,
    "W": logging.WARNING,
    "WARNING": logging.WARNING,
    "I": logging.INFO,
    "INFO": logging.INFO,
    "D": logging.DEBUG,
    "DEBUG": logging.DEBUG,
}

# Keepalive constants
_KEEPALIVE_INTERVAL_SEC = 30
_KEEPALIVE_MAX_CONSECUTIVE_FAILURES = 20  # be very generous
_KEEPALIVE_MAX_DURATION_SEC = 24 * 60 * 60  # 24 hours
_KEEPALIVE_WARNING_BEFORE_EXIT_SEC = 30 * 60  # 30 minutes


def set_logging_level() -> None:
    log_level = click.get_current_context().obj["log"]
    if isinstance(log_level, str):
        level = _log_level_str_to_level[log_level]
    else:
        level = log_level

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if log_level == _HUMANFRIENDLY_LOG_LEVEL:
        rich_handler = rich.logging.RichHandler(
            show_time=False, show_level=False, show_path=False
        )
    else:
        # Rich handler adds time, levels, file location etc.
        rich_handler = rich.logging.RichHandler()

    root_logger.addHandler(rich_handler)
    # Enable deprecation warnings raised in this module.
    warnings.filterwarnings(
        "default", category=DeprecationWarning, module=r"^truss\.cli\\b"
    )


def check_is_interactive() -> bool:
    """Detects if CLI is operated interactively by human, so we can ask things,
    that we would want to skip for automated subprocess/CI contexts."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _store_param_callback(ctx: click.Context, param: click.Parameter, value: str):
    # We use this for params that are not "exposed" to the signature of the subcommands.
    # therefore we store them directly on the context (not in contex.params).
    ctx.ensure_object(dict)[param.name] = value


def get_required_option(ctx: click.Context, name: str) -> object:
    value = ctx.find_root().obj.get(name)
    if value is None:
        raise RuntimeError(
            f"Required option '{name}' was not set in click context. "
            "This is a bug, all commands must use `common_options` decorator."
        )
    return value


def log_level_option(f: Callable[..., object]) -> Callable[..., object]:
    return click.option(
        "--log",
        default=_HUMANFRIENDLY_LOG_LEVEL,
        help="Customizes logging.",
        type=click.Choice(list(_log_level_str_to_level.keys()), case_sensitive=False),
        callback=_store_param_callback,
        expose_value=False,
    )(f)


def _non_interactive_option(f: Callable[..., object]) -> Callable[..., object]:
    return click.option(
        "--non-interactive",
        is_flag=True,
        default=False,
        help="Disables interactive prompts, use in CI / automated execution contexts.",
        expose_value=False,
        callback=_store_param_callback,
    )(f)


def _error_handling(f: Callable[..., object]) -> Callable[..., object]:
    @wraps(f)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            f(*args, **kwargs)
        except click.UsageError as e:
            raise e
        except Exception as e:
            ctx = click.get_current_context()
            log_level = get_required_option(ctx, "log")
            escaped_e = escape(str(e))
            if log_level == _HUMANFRIENDLY_LOG_LEVEL:
                console.print(
                    f"[bold red]ERROR {type(e).__name__}[/bold red]: {escaped_e}",
                    highlight=True,
                )
            else:
                console.print_exception(show_locals=True)

            if isinstance(e, pydantic.ValidationError):
                console.print(
                    "[bold yellow]In case of 'ValidationErrors' there are two common issues:[/bold yellow]\n"
                    "[yellow]"
                    " * 'Extra inputs are not permitted...': using a new 'TrussConfig' "
                    "field that is not yet in your local truss CLI version -> upgrade truss version.\n"
                    " * 'Input should be ...': using muddy types, e.g. a float where a string "
                    "is expected -> check the exact message above and fix.[/yellow]",
                    highlight=True,
                )

            ctx.exit(1)

    return wrapper


def upgrade_dialogue():
    try:
        self_upgrade.notify_if_outdated(truss.__version__)
    except Exception as e:
        logger.debug(f"Upgrade check failed: {e}")


def common_options(
    add_middleware: bool = True,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    def decorator(f: Callable[..., object]) -> Callable[..., object]:
        @wraps(f)
        @log_level_option
        @_non_interactive_option
        @_error_handling
        def wrapper(*args: object, **kwargs: object) -> Any:
            if add_middleware:
                set_logging_level()
                upgrade_dialogue()

            return f(*args, **kwargs)

        return wrapper

    return decorator


def format_link(url: str, display_text: Optional[str] = None) -> str:
    display_text = display_text or url
    return f"[link={url}]{display_text}[/link]"


def is_human_log_level(ctx: click.Context) -> bool:
    return get_required_option(ctx, "log") != _HUMANFRIENDLY_LOG_LEVEL


def _normalize_iso_timestamp(iso_timestamp: str) -> str:
    iso_timestamp = iso_timestamp.strip()
    if iso_timestamp.endswith("Z"):
        iso_timestamp = iso_timestamp[:-1] + "+00:00"

    tz_part = ""
    tz_match = re.search(r"([+-]\d{2}:\d{2}|[+-]\d{4})$", iso_timestamp)
    if tz_match:
        tz_part = tz_match.group(0)
        iso_timestamp = iso_timestamp[: tz_match.start()]

    iso_timestamp = iso_timestamp.rstrip()

    if tz_part and ":" not in tz_part:
        tz_part = f"{tz_part[:3]}:{tz_part[3:]}"

    fractional_match = re.search(r"\.(\d+)$", iso_timestamp)
    if fractional_match:
        fractional_digits = fractional_match.group(1)
        if len(fractional_digits) > 6:
            iso_timestamp = (
                iso_timestamp[: fractional_match.start()] + "." + fractional_digits[:6]
            )

    return f"{iso_timestamp}{tz_part}"


# NOTE: `pyproject.toml` declares support down to Python 3.9, whose
# `datetime.fromisoformat` cannot parse nanosecond fractions or colonless offsets,
# so normalize timestamps before parsing.
def format_localized_time(iso_timestamp: str) -> str:
    try:
        utc_time = datetime.datetime.fromisoformat(iso_timestamp)
    except ValueError:
        # Handle non-standard formats (nanoseconds, Z suffix, colonless offsets)
        normalized_timestamp = _normalize_iso_timestamp(iso_timestamp)
        utc_time = datetime.datetime.fromisoformat(normalized_timestamp)

    local_time = utc_time.astimezone()
    return local_time.strftime("%Y-%m-%d %H:%M:%S")


def format_bytes_to_human_readable(bytes: int) -> str:
    if bytes > 1000 * 1000 * 1000 * 1000:
        return f"{bytes / (1000 * 1000 * 1000 * 1000):.2f} TB"
    if bytes > 1000 * 1000 * 1000:
        return f"{bytes / (1000 * 1000 * 1000):.2f} GB"
    elif bytes > 1000 * 1000:
        return f"{bytes / (1000 * 1000):.2f} MB"
    elif bytes > 1000:
        return f"{bytes / 1000:.2f} KB"
    else:
        return f"{bytes} B"


def wait_for_development_model_ready(
    model_hostname: str,
    model_id: str,
    dev_version_id: str,
    remote_provider: BasetenRemote,
    console: "rich_console.Console",
    api_key: str,
) -> None:
    # Wake the model in case it's scaled to zero
    wake_url = f"{model_hostname}/development/wake"
    headers = {"Authorization": f"Api-Key {api_key}"}
    try:
        requests_lib.post(wake_url, headers=headers, timeout=10)
    except requests_lib.RequestException:
        # best effort
        pass

    # Wait for model to be ready before starting keepalive
    with console.status(
        "[bold green]Waiting for development model to be ready..."
    ) as status:
        while True:
            time.sleep(1)
            try:
                deployment = remote_provider.api.get_deployment(
                    model_id, dev_version_id
                )
                deployment_status = deployment["status"]
            except Exception:
                continue
            status.update(
                f"[bold green]Waiting for development model to be ready... "
                f"Current Status: {deployment_status}"
            )
            if deployment_status in [ACTIVE_STATUS, "LOADING_MODEL"]:
                break
            if deployment_status not in DEPLOYING_STATUSES + [
                "SCALED_TO_ZERO",
                "WAKING_UP",
                "UPDATING",
            ]:
                console.print(
                    f"❌ Development model failed with status {deployment_status}.",
                    style="red",
                )
                sys.exit(1)


def keepalive_loop(
    model_hostname: str, api_key: str, stop_event: threading.Event
) -> None:
    headers = {"Authorization": f"Api-Key {api_key}"}
    consecutive_failures = 0
    start_time = time.time()
    keepalive_url = f"{model_hostname}/development/sync/v1/models/model"
    warning_emitted = False

    while not stop_event.is_set():
        elapsed_time = time.time() - start_time

        if time.time() - start_time > _KEEPALIVE_MAX_DURATION_SEC:
            console.print(
                "⚠️  Keepalive has been running for 24 hours. Exiting truss watch.",
                style="yellow",
            )
            os._exit(0)

        # emit warning before exit once
        if not warning_emitted and elapsed_time > (
            _KEEPALIVE_MAX_DURATION_SEC - _KEEPALIVE_WARNING_BEFORE_EXIT_SEC
        ):
            console.print(
                "⚠️  Keepalive will automatically exit in 30 minutes (24 hour limit).",
                style="yellow",
            )
            warning_emitted = True

        try:
            resp = requests_lib.get(keepalive_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                consecutive_failures = 0
            elif 400 <= resp.status_code < 500:
                # Ignore 4xx errors
                pass
            else:
                # Count 5xx errors as failures
                consecutive_failures += 1
        except requests_lib.RequestException:
            consecutive_failures += 1

        if consecutive_failures >= _KEEPALIVE_MAX_CONSECUTIVE_FAILURES:
            console.print(
                f"⚠️  Keepalive ping failed {consecutive_failures} times in a row. "
                "Exiting truss watch.",
                style="red",
            )
            os._exit(1)  # kill process not just the thread

        stop_event.wait(timeout=_KEEPALIVE_INTERVAL_SEC)
