from fastapi import FastAPI

import torch

app = FastAPI()

@app.get("/health")
async def root():
    return {"message": "OK"}

@app.post("/predict")
async def root():
    return {"message": "Hello World", "is_torch_cuda_available": torch.cuda.is_available()}
