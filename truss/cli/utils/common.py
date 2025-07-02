import datetime
import logging
import sys
import warnings
from functools import wraps
from typing import Any, Callable, Optional

import pydantic
import rich
import rich.live
import rich.logging
import rich.spinner
import rich.table
import rich.traceback
import rich_click as click
from rich.markup import escape

import truss
from truss.cli.utils import self_upgrade
from truss.cli.utils.output import console
from truss.util import user_config

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
    ctx = click.get_current_context()
    if (
        not get_required_option(ctx, "non_interactive")
        and check_is_interactive()
        and user_config.settings.enable_auto_upgrade
    ):
        self_upgrade.upgrade_dialogue(truss.__version__)


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


def format_localized_time(iso_timestamp: str) -> str:
    if iso_timestamp.endswith("Z"):
        iso_timestamp = iso_timestamp.replace("Z", "+00:00")
    utc_time = datetime.datetime.fromisoformat(iso_timestamp)
    local_time = utc_time.astimezone()
    return local_time.strftime("%Y-%m-%d %H:%M")
