from pathlib import Path

# If changing model repo path, please updated inside tensorrt_llm config.pbtxt as well
TENSORRT_LLM_MODEL_REPOSITORY_PATH = Path("/packages/tensorrt_llm_model_repository/")
GRPC_SERVICE_PORT = 8001
HTTP_SERVICE_PORT = 8003
HF_AUTH_KEY_CONSTANT = "HUGGING_FACE_HUB_TOKEN"
TOKENIZER_KEY_CONSTANT = "TRITON_TOKENIZER_REPOSITORY"
ENTRYPOINT_MODEL_NAME = "ensemble"
