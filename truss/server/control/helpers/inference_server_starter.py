import os

import requests
from tenacity import Retrying, stop_after_attempt, wait_exponential


def inference_server_startup_flow(application):
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
    """
    inference_server_controller = application.config["inference_server_controller"]
    patch_ping_url = os.environ.get("PATCH_PING_URL_TRUSS")
    if patch_ping_url is None:
        inference_server_controller.start()
        return

    truss_hash = inference_server_controller.truss_hash()
    payload = {"truss_hash": truss_hash}

    for attempt in Retrying(
        stop=stop_after_attempt(15),
        wait=wait_exponential(multiplier=2, min=1, max=4),
    ):
        with attempt:
            try:
                application.logger.info(
                    f"Pinging {patch_ping_url} for patch with hash {truss_hash}"
                )
                resp = requests.post(patch_ping_url, json=payload)
                resp.raise_for_status()
                resp_body = resp.json()

                # If hash is current start inference server, otherwise delay that
                # for when patch is applied.
                if "is_current" in resp_body and resp_body["is_current"] is True:
                    application.logger.info(
                        "Hash is current, starting inference server"
                    )
                    inference_server_controller.start()
            except Exception as exc:  # noqa
                application.logger.warning(
                    f"Patch ping attempt failed with error {exc}"
                )
                raise exc
