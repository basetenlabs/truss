import os
from logging import Logger

import requests
from anyio import to_thread
from helpers.inference_server_controller import InferenceServerController
from tenacity import Retrying, stop_after_attempt, wait_exponential

PATCH_PING_MAX_ATTEMPTS = 15
NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ProxyError,
    requests.exceptions.Timeout,
)


def _classify_error(exc: BaseException) -> str:
    if isinstance(exc, NETWORK_EXCEPTIONS):
        return "network error"
    return "error"


def inference_server_startup_flow(
    inference_server_controller: InferenceServerController, logger: Logger
) -> None:
    """
    Perform the inference server startup flow

    Inference server startup flow supports checking for patches. If a patch ping
    url is provided then we hit that url to start the sync mechanism. The ping
    calls with current truss hash. The patch ping endpoint should return a
    response indicating, either that the supplied hash is current or that the
    request has been accepted. Acceptance of request means that a patch will be
    supplied soon to the truss (by calling of /control/patch endpoint).

    If we find that our hash is current, we start the inference server
    immediately. Otherwise, we delay the start to when the patch is supplied.

    The goal is to start the inference server as soon as we have the latest
    code, but not before.
    Example responses:
    {"is_current": true}
    {"accepted": true}

    Exceptions are caught and logged here rather than propagated. This is
    invoked via asyncio.create_task() with no awaiter, so a raised exception
    would only surface as Python's "Task exception was never retrieved"
    warning at GC time. Logging a clear summary line at the failure boundary
    is more actionable than that warning.
    """
    patch_ping_url = os.environ.get("PATCH_PING_URL_TRUSS")
    if patch_ping_url is None:
        inference_server_controller.start()
        return

    truss_hash = inference_server_controller.truss_hash()
    payload = {"truss_hash": truss_hash}
    logger.info(f"Pinging {patch_ping_url} for patch with hash {truss_hash}")

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(PATCH_PING_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=2, min=1, max=4),
            reraise=True,
        ):
            with attempt:
                try:
                    resp = requests.post(patch_ping_url, json=payload)
                    resp.raise_for_status()
                    resp_body = resp.json()

                    # If hash is current start inference server, otherwise delay
                    # that for when patch is applied.
                    if "is_current" in resp_body and resp_body["is_current"] is True:
                        logger.info("Hash is current, starting inference server")
                        inference_server_controller.start()
                except Exception as exc:
                    attempt_number = attempt.retry_state.attempt_number
                    if attempt_number == 1:
                        logger.warning(
                            f"Patch ping {_classify_error(exc)}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        logger.info(
                            f"Patch ping retry "
                            f"{attempt_number}/{PATCH_PING_MAX_ATTEMPTS}: "
                            f"{type(exc).__name__}"
                        )
                    raise
    except Exception as exc:
        logger.error(
            f"Patch ping failed after {PATCH_PING_MAX_ATTEMPTS} attempts; "
            f"inference server will not start. "
            f"Last {_classify_error(exc)} reaching {patch_ping_url}: "
            f"{type(exc).__name__}: {exc}"
        )


async def async_inference_server_startup_flow(
    inference_server_controller: InferenceServerController, logger: Logger
) -> None:
    # Exceptions from the patch ping flow are caught and logged inside
    # inference_server_startup_flow; this coroutine never raises from that path.
    return await to_thread.run_sync(
        inference_server_startup_flow, inference_server_controller, logger
    )
