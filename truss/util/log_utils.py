import logging
import threading
from typing import List, Optional


# TODO: add test.
class LogInterceptor(logging.Handler):
    """This context manager intercepts logs at root level and allows to retrieve them
    later. It uses the formatter of the first root handler (if present).

    Specifically it allows different threads to each have an instance of this context
    manager and keeping the logs separated between the threads.
    Note that this is different from a single instance being used by multiple threads.
    """

    _formatter: Optional[logging.Formatter]
    _original_handlers: List[logging.Handler]
    _log_messages: List[str]
    _thread_local = threading.local()

    def __init__(self) -> None:
        super().__init__()
        self._formatter = None
        self._log_messages = []
        self._original_handlers = []

    def __enter__(self) -> "LogInterceptor":
        if not hasattr(LogInterceptor._thread_local, "handlers"):
            LogInterceptor._thread_local.handlers = []
        LogInterceptor._thread_local.handlers.append(self)
        self._original_handlers = logging.root.handlers[:]
        logging.root.handlers = [self]
        self._formatter = (
            logging.root.handlers[0].formatter if logging.root.handlers else None
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        logging.root.handlers = self._original_handlers
        LogInterceptor._thread_local.handlers.pop()

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(LogInterceptor._thread_local, "handlers"):
            current_handler = LogInterceptor._thread_local.handlers[-1]
            if self._formatter:
                formatted_record = self._formatter.format(record)
            else:
                formatted_record = self.format(record)
            current_handler._log_messages.append(formatted_record)

    def get_logs(self) -> List[str]:
        return self._log_messages
