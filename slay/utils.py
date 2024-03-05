import contextlib
import io
import json
import logging

import pydantic


@contextlib.contextmanager
def log_level(level: int):
    current_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(current_logging_level)


@contextlib.contextmanager
def no_print():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def pydantic_pp(model: pydantic.BaseModel) -> str:
    return json.dumps(model.model_dump(), indent=4)
