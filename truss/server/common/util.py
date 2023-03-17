import logging
import sys

from pythonjsonlogger import jsonlogger

LEVEL = logging.INFO

JSON_LOG_HANDLER = logging.StreamHandler(stream=sys.stderr)
JSON_LOG_HANDLER.set_name("json_logger_handler")
JSON_LOG_HANDLER.setLevel(LEVEL)
JSON_LOG_HANDLER.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
)


def model_supports_predict_proba(model: object) -> bool:
    if not hasattr(model, "predict_proba"):
        return False
    if hasattr(
        model, "_check_proba"
    ):  # noqa eg Support Vector Machines *can* predict proba if they made certain choices while training
        try:
            model._check_proba()
            return True
        except AttributeError:
            return False
    return True


def assign_request_to_inputs_instances_after_validation(body: dict) -> dict:
    # we will treat "instances" and "inputs" the same
    if "instances" in body and "inputs" not in body:
        body["inputs"] = body["instances"]
    elif "inputs" in body and "instances" not in body:
        body["instances"] = body["inputs"]
    return body


def setup_logging():
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
