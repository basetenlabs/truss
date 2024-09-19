import os
from fastapi import FastAPI

import torch

app = FastAPI()

@app.get("/health")
async def root():
    return {"message": "OK"}

@app.post("/predict")
async def root():
    return {
        "message": "Hello World",
        "is_torch_cuda_available": torch.cuda.is_available(),
        "hf_token_from_env": os.environ.get("HF_TOKEN"),
        "is_secret_mounted": os.path.exists("/secrets/hf_access_token"),
        }
