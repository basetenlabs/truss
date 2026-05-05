"""LLMMock — minimal CPU-only HTTP-fronted Truss that mocks an LLM completion.

POST /predict with `{"prompt": "..."}` → returns `{"completion": "echo: ..."}`.
Stands in for the real OpenAI-compatible LLM in the FDE voice agent without
the OpenAI SDK / TRT-LLM weight overhead.
"""


class Model:
    def __init__(self, **kwargs) -> None:
        pass

    def predict(self, request: dict) -> dict:
        prompt = str(request.get("prompt", ""))
        return {"completion": f"echo: {prompt}"}
