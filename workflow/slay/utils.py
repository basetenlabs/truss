import contextlib
import logging


@contextlib.contextmanager
def log_level(level: int):
    current_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(current_logging_level)
