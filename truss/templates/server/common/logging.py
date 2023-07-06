import logging
import sys

from pythonjsonlogger import jsonlogger

LEVEL: int = logging.INFO


class APIJsonFormatter(jsonlogger.JsonFormatter):
    """
    A custom JsonFormatter to reformat the web server log entries.
    """

    def process_log_record(self, log_record):
        message = log_record.get("message", "")

        try:
            # Parse the necessary information from the message
            parts = message.split(" ")
            request_type = parts[2][1:]
            endpoint = parts[3].split("%3A")[1]
            status_code = parts[5]

            status_codes = {
                "200": "OK",
                "404": "NOT_FOUND",
                "500": "INTERNAL_ERROR",
                "503": "SERVICE_UNAVAILABLE"
                # Add more status codes and their corresponding descriptions if needed
            }

            new_message = f"{request_type} /{endpoint} {status_code} {status_codes.get(status_code)}"
        except IndexError:
            new_message = message

        # Replace 'message' field in the log record
        log_record["message"] = new_message

        return super().process_log_record(log_record)


JSON_LOG_HANDLER = logging.StreamHandler(stream=sys.stdout)
JSON_LOG_HANDLER.set_name("json_logger_handler")
JSON_LOG_HANDLER.setLevel(LEVEL)
JSON_LOG_HANDLER.setFormatter(APIJsonFormatter("%(asctime)s %(levelname)s %(message)s"))


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # for any health check endpoints, lets skip logging
        return (
            record.getMessage().find("GET / ") == -1
            and record.getMessage().find("GET /v1/models/model ") == -1
        )


def setup_logging() -> None:
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]

    for logger in loggers:
        logger.setLevel(LEVEL)
        logger.propagate = False

        setup = False

        # let's not thrash the handlers unnecessarily
        for handler in logger.handlers:
            if handler.name == JSON_LOG_HANDLER.name:
                setup = True

        if not setup:
            logger.handlers.clear()
            logger.addHandler(JSON_LOG_HANDLER)

        # some special handling for request logging
        if logger.name == "uvicorn.access":
            logger.addFilter(HealthCheckFilter())
