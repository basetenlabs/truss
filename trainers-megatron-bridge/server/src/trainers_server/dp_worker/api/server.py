from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from trainers_server.shared.models import (
    AdamParams,
    ForwardBackwardDetails,
    ForwardBackwardResult,
    OptimStepDetails,
    OptimStepResult,
    SampleDetails,
    SampleResult,
    SaveStateDetails,
    SaveStateResult,
    ToInferenceResult,
)

from .controller import RLController
from .models import (
    RLControllerConfig,
    StatusResult,
)


def _run_or_raise(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def create_app(config: Optional[RLControllerConfig] = None, *, controller: Optional[RLController] = None) -> FastAPI:
    app = FastAPI(title="RL Training API")
    if controller is None:
        controller = RLController(config or RLControllerConfig())
    app.state.rl_controller = controller

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.post("/forward_backward", response_model=ForwardBackwardResult)
    async def forward_backward(details: ForwardBackwardDetails) -> ForwardBackwardResult:
        return _run_or_raise(controller.forward_backward, details)

    @app.post("/optim_step", response_model=OptimStepResult)
    async def optim_step(details: OptimStepDetails = OptimStepDetails(adam_params=AdamParams())) -> OptimStepResult:
        return _run_or_raise(controller.optim_step, details)

    @app.post("/to_inference", response_model=ToInferenceResult)
    async def to_inference() -> ToInferenceResult:
        return _run_or_raise(controller.to_inference)

    @app.post("/to_training", response_model=StatusResult)
    async def to_training() -> StatusResult:
        return _run_or_raise(controller.to_training)

    @app.post("/sample", response_model=SampleResult)
    async def sample(details: SampleDetails) -> SampleResult:
        return _run_or_raise(controller.sample, details)

    @app.post("/save_state", response_model=SaveStateResult)
    async def save_state(details: SaveStateDetails) -> SaveStateResult:
        return _run_or_raise(controller.save_state, details.path)

    @app.get("/status", response_model=StatusResult)
    async def get_status() -> StatusResult:
        return _run_or_raise(controller.get_status)

    return app
