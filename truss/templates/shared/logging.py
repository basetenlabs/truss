import logging
import sys

from pythonjsonlogger import jsonlogger

LEVEL: int = logging.INFO

JSON_LOG_HANDLER = logging.StreamHandler(stream=sys.stdout)
JSON_LOG_HANDLER.set_name("json_logger_handler")
JSON_LOG_HANDLER.setLevel(LEVEL)
JSON_LOG_HANDLER.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
)


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # for any health check endpoints, lets skip logging
        return (
            record.getMessage().find("GET / ") == -1
            and record.getMessage().find("GET /v1/models/model ") == -1
        )


class StreamToLogger:
    """
    StreamToLogger redirects stdout and stderr to logger
    """

    def __init__(self, logger, log_level, stream):
        self.logger = logger
        self.log_level = log_level
        self.stream = stream

    def __getattr__(self, name):
        # we need to pass `isatty` from the stream for uvicorn
        # this is a more general, less hacky fix
        return getattr(self.stream, name)

    def write(self, buf):
        self.logger.log(self.log_level, buf)

    def flush(self):
        """
        This is a no-op function. It only exists to prevent
        AttributeError in case some part of the code attempts to call flush()
        on instances of StreamToLogger. Thus, we define this method as a safety
        measure.
        """
        pass


def setup_logging() -> None:
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]

    sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO, sys.__stdout__)  # type: ignore
    sys.stderr = StreamToLogger(logging.getLogger(), logging.INFO, sys.__stderr__)  # type: ignore

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
