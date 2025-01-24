import asyncio
from typing import Any, Dict

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from helpers.errors import ModelLoadFailed, ModelNotReady
from httpx import URL, ConnectError, RemoteProtocolError
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response
from tenacity import RetryCallState, Retrying, retry_if_exception_type, wait_fixed

INFERENCE_SERVER_START_WAIT_SECS = 60

control_app = APIRouter()


@control_app.get("/")
def index():
    return {}


async def proxy(request: Request):
    inference_server_process_controller = (
        request.app.state.inference_server_process_controller
    )
    client: httpx.AsyncClient = request.app.state.proxy_client

    path = _reroute_if_health_check(request.url.path)
    url = URL(path=path, query=request.url.query.encode("utf-8"))

    # 2 min connect timeouts, no timeout for requests.
    # We don't want requests to fail due to timeout on the proxy
    timeout = httpx.Timeout(None, connect=2 * 60.0)
    try:
        request_body = await request.body()
    except ClientDisconnect:
        # If the client disconnects, we don't need to proxy the request
        return Response(status_code=499)

    inf_serv_req = client.build_request(
        request.method,
        url,
        headers=request.headers.raw,
        content=request_body,
        timeout=timeout,
    )

    # Wait a bit for inference server to start
    for attempt in Retrying(
        retry=(
            retry_if_exception_type(ConnectError)
            | retry_if_exception_type(ModelNotReady)
            | retry_if_exception_type(RemoteProtocolError)
            | retry_if_exception_type(httpx.ReadError)
            | retry_if_exception_type(httpx.ReadTimeout)
            | retry_if_exception_type(httpx.ConnectTimeout)
        ),
        stop=_custom_stop_strategy,
        wait=wait_fixed(1),
    ):
        with attempt:
            try:
                if inference_server_process_controller.is_inference_server_intentionally_stopped():
                    raise ModelLoadFailed("Model load failed")
                resp = await client.send(inf_serv_req, stream=True)

                if await _is_model_not_ready(resp):
                    raise ModelNotReady("Model has started running, but not ready yet.")
            except (RemoteProtocolError, ConnectError) as exp:
                # This check is a bit expensive so we don't do it before every request, we
                # do it only if request fails with connection error. If the inference server
                # process is running then we continue waiting for it to start (by retrying),
                # otherwise we bail.
                if (
                    inference_server_process_controller.inference_server_ever_started()
                    and not inference_server_process_controller.is_inference_server_running()
                ):
                    error_msg = (
                        "It appears your model has stopped running. This often means' \
                        ' it crashed and may need a fix to get it running again."
                    )
                    return JSONResponse(error_msg, 503)
                raise exp

    if _is_streaming_response(resp):
        return StreamingResponse(
            resp.aiter_bytes(), media_type="application/octet-stream"
        )

    await resp.aread()
    response = Response(resp.content, resp.status_code, resp.headers)
    return response


control_app.add_route("/v1/{path:path}", proxy, ["GET", "POST"])


@control_app.post("/control/patch")
async def patch(request: Request) -> Dict[str, str]:
    request.app.state.logger.info("Patch request received.")
    patch_request = await request.json()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: request.app.state.inference_server_controller.apply_patch(
            patch_request
        ),
    )
    request.app.state.logger.info("Patch applied successfully")
    return {"msg": "Patch applied successfully"}


@control_app.get("/control/truss_hash")
async def truss_hash(request: Request) -> Dict[str, Any]:
    t_hash = request.app.state.inference_server_controller.truss_hash()
    return {"result": t_hash}


@control_app.post("/control/restart_inference_server")
async def restart_inference_server(request: Request) -> Dict[str, str]:
    request.app.state.inference_server_controller.restart()

    return {"msg": "Inference server started successfully"}


@control_app.get("/control/has_partially_applied_patch")
async def has_partially_applied_patch(request: Request) -> Dict[str, Any]:
    app_has_partially_applied_patch = (
        request.app.state.inference_server_controller.has_partially_applied_patch()
    )
    return {"result": app_has_partially_applied_patch}


@control_app.post("/control/stop_inference_server")
async def stop_inference_server(request: Request) -> Dict[str, str]:
    request.app.state.inference_server_controller.stop()
    return {"msg": "Inference server stopped successfully"}


async def _is_model_not_ready(resp) -> bool:
    if resp.status_code == 503:
        await resp.aread()
        return resp.content is not None and "model is not ready" in resp.content.decode(
            "utf-8"
        )
    return False


def _is_streaming_response(resp) -> bool:
    for header_name, value in resp.headers.items():
        if header_name.lower() == "transfer-encoding" and value.lower() == "chunked":
            return True
    return False


def _reroute_if_health_check(path: str) -> str:
    """
    Reroutes calls from the Operator to the inference server's health check endpoint (/v1/models/model) to /v1/models/model/loaded instead.
    This is done to avoid running custom health checks when the Operator is checking if the inference server is ready.
    """
    if path == "/v1/models/model":
        path = "/v1/models/model/loaded"
    return path


def _custom_stop_strategy(retry_state: RetryCallState) -> bool:
    # Stop after 10 attempts for ModelNotReady
    if retry_state.outcome is not None and isinstance(
        retry_state.outcome.exception(), ModelNotReady
    ):
        # Check if the retry limit for ModelNotReady has been reached
        return retry_state.attempt_number >= 10
    # For all other exceptions, stop after INFERENCE_SERVER_START_WAIT_SECS
    seconds_since_start = (
        retry_state.seconds_since_start
        if retry_state.seconds_since_start is not None
        else 0.0
    )
    return seconds_since_start >= INFERENCE_SERVER_START_WAIT_SECS
