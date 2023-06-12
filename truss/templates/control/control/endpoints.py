from typing import Any, Dict

import requests
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from helpers.errors import ModelLoadFailed, ModelNotReady
from requests.exceptions import ConnectionError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

INFERENCE_SERVER_START_WAIT_SECS = 60


control_app = APIRouter()


@control_app.get("/")
def index():
    return {}


@control_app.get("/v1/{full_path:path}")
@control_app.post("/v1/{full_path:path}")
async def proxy(full_path: str, request: Request):
    inference_server_port = request.app.state.inference_server_port
    inference_server_process_controller = (
        request.app.state.inference_server_process_controller
    )

    # Wait a bit for inference server to start
    for attempt in Retrying(
        retry=(
            retry_if_exception_type(ConnectionError)
            | retry_if_exception_type(ModelNotReady)
        ),
        stop=stop_after_attempt(INFERENCE_SERVER_START_WAIT_SECS),
        wait=wait_fixed(1),
    ):
        with attempt:
            try:
                if (
                    inference_server_process_controller.is_inference_server_intentionally_stopped()
                ):
                    raise ModelLoadFailed("Model load failed")

                resp = requests.request(
                    method=request.method,
                    url=f"http://localhost:{inference_server_port}/v1/{full_path}",
                    data=await request.body(),
                    cookies=request.cookies,
                    headers=request.headers,
                )
                if _is_model_not_ready(resp):
                    raise ModelNotReady("Model has started running, but not ready yet.")
            except ConnectionError as exp:
                # This check is a bit expensive so we don't do it before every request, we
                # do it only if request fails with connection error. If the inference server
                # process is running then we continue waiting for it to start (by retrying),
                # otherwise we bail.
                if (
                    inference_server_process_controller.inference_server_ever_started()
                    and not inference_server_process_controller.is_inference_server_running()
                ):
                    error_msg = "It appears your model has stopped running. This often means' \
                        ' it crashed and may need a fix to get it running again."
                    return JSONResponse(error_msg, 503)
                raise exp

    response = Response(resp.content, resp.status_code, resp.headers)
    return response


@control_app.post("/control/patch")
async def patch(request: Request) -> Dict[str, str]:
    request.app.state.logger.info("Patch request received.")
    patch_request = await request.json()
    request.app.state.inference_server_controller.apply_patch(patch_request)
    request.app.state.logger.info("Patch applied successfully")
    return {"msg": "Patch applied successfully"}


@control_app.get("/control/truss_hash")
def truss_hash(request: Request) -> Dict[str, Any]:
    t_hash = request.app.state.inference_server_controller.truss_hash()
    return {"result": t_hash}


@control_app.post("/control/restart_inference_server")
def restart_inference_server(request: Request) -> Dict[str, str]:
    request.app.state.inference_server_controller.restart()

    return {"msg": "Inference server started successfully"}


@control_app.get("/control/has_partially_applied_patch")
def has_partially_applied_patch(request: Request) -> Dict[str, Any]:
    app_has_partially_applied_patch = (
        request.app.state.inference_server_controller.has_partially_applied_patch()
    )
    return {"result": app_has_partially_applied_patch}


@control_app.post("/control/stop_inference_server")
def stop_inference_server(request: Request) -> Dict[str, str]:
    request.app.state.inference_server_controller.stop()
    return {"msg": "Inference server stopped successfully"}


def _is_model_not_ready(resp) -> bool:
    return (
        resp.status_code == 503
        and resp.content is not None
        and "model is not ready" in resp.content.decode("utf-8")
    )
