import os

import torch
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health():
    return {"message": "OK"}


@app.post("/predict")
async def root():
    return {
        "message": "Hello World",
        "is_torch_cuda_available": torch.cuda.is_available(),
        "is_env_var_passed": os.environ.get("HF_TOKEN") is not None,
        "is_secret_mounted": os.path.exists("/secrets/hf_access_token"),
    }
