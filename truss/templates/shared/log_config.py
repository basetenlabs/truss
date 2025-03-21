import logging
import os
import urllib.parse
from typing import Any, Mapping

from pythonjsonlogger import jsonlogger

LOCAL_DATE_FORMAT = "%H:%M:%S"


def _disable_json_logging() -> bool:
    return bool(os.environ.get("DISABLE_JSON_LOGGING"))


class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        excluded_paths = {
            "GET / ",
            "GET /v1/models/model ",
            "GET /v1/models/model/loaded ",
        }
        msg = record.getMessage()
        return not any(path in msg for path in excluded_paths)


class _WebsocketOpenFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # There is already the line
        # `('172.17.0.1', 54024) - "WebSocket /v1/websocket" [accepted]`
        # So we filter this additional log for open.
        return "connection open" not in msg


class _MetricsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/metrics" not in record.getMessage()


class _AccessJsonFormatter(jsonlogger.JsonFormatter):
    def format(self, record: logging.LogRecord) -> str:
        # Uvicorn sets record.msg = '%s - "%s %s HTTP/%s" %d' and
        # record.args = (addr, method, path, version, status).
        # Python's logging system resolves final
        # record.message = record.msg % record.args unless we override record.msg.
        if record.name == "uvicorn.access" and record.args and len(record.args) == 5:
            client_addr, method, raw_path, version, status = record.args
            path_decoded = urllib.parse.unquote(str(raw_path))
            new_message = (
                f"Handled request: {method} {path_decoded} HTTP/{version} {status}"
            )
            record.msg = new_message
            record.args = ()  # Ensure Python doesn't reapply the old format string
        return super().format(record)


class _AccessFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.name == "uvicorn.access" and record.args and len(record.args) == 5:
            client_addr, method, raw_path, version, status = record.args
            path_decoded = urllib.parse.unquote(str(raw_path))
            new_message = (
                f"Handled request - {method} {path_decoded} HTTP/{version} {status}"
            )
            record.msg = new_message
            record.args = ()
        return super().format(record)


def make_log_config(log_level: str) -> Mapping[str, Any]:
    # Warning: `ModelWrapper` depends on correctly setup `uvicorn` logger,
    # if you change/remove that logger, make sure `ModelWrapper` has a suitable
    # alternative logger that is also correctly setup in the load thread.
    formatters = (
        {
            "default_formatter": {
                "format": "%(asctime)s.%(msecs)04d %(levelname)s %(message)s",
                "datefmt": LOCAL_DATE_FORMAT,
            },
            "access_formatter": {
                "()": _AccessFormatter,
                "format": "%(asctime)s.%(msecs)04d %(levelname)s %(message)s",
                "datefmt": LOCAL_DATE_FORMAT,
            },
        }
        if _disable_json_logging()
        else {
            "default_formatter": {
                "()": jsonlogger.JsonFormatter,
                "format": "%(asctime)s %(levelname)s %(message)s",
            },
            "access_formatter": {
                "()": _AccessJsonFormatter,
                "format": "%(asctime)s %(levelname)s %(message)s",
            },
        }
    )

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "health_check_filter": {"()": _HealthCheckFilter},
            "websocket_filter": {"()": _WebsocketOpenFilter},
            "metrics_filter": {"()": _MetricsFilter},
        },
        "formatters": formatters,
        "handlers": {
            "default_handler": {
                "formatter": "default_formatter",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access_handler": {
                "formatter": "access_formatter",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default_handler"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default_handler"],
                "level": "INFO",
                "propagate": False,
                # For some reason websockets use error logger.
                "filters": ["websocket_filter"],
            },
            "uvicorn.access": {
                "handlers": ["access_handler"],
                "level": "INFO",
                "propagate": False,
                "filters": ["health_check_filter", "metrics_filter"],
            },
            "httpx": {
                "handlers": ["default_handler"],
                "level": "INFO",
                "propagate": False,
                "filters": ["metrics_filter"],
            },
        },
        # Catch-all for module loggers
        "root": {"handlers": ["default_handler"], "level": log_level},
    }
    return log_config
