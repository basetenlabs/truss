import asyncio
from typing import Any, Dict

from fastapi import APIRouter
from starlette.requests import Request

INFERENCE_SERVER_START_WAIT_SECS = 60


control_app = APIRouter()


@control_app.get("/")
def index():
    return {}


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
