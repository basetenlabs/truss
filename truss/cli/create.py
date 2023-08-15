from InquirerPy import inquirer
from truss.truss_config import Build, ModelServer

REQUIRED_ARGUMENTS = {
    ModelServer.TGI: {
        "model_id": {
            "meta-llama/Llama-2-7b-chat-hf",
            "bigcode/starcoder",
            "google/flan-t5-xxl",
            "EleutherAI/gpt-neox-20b",
            "tiiuae/falcon-7b",
        },
        "endpoint": {"generate_stream", "generate"},
        # "dtype": {"bfloat16", "float16"},
    },
    ModelServer.VLLM: {
        "model": {
            "meta-llama/Llama-2-7b-chat-hf",
            "tiiuae/falcon-7b",
            # TODO(abu): Add more models
        },
        "endpoint": {"Completions", "ChatCompletions"},
    },
}


def select_server_backend(server_backend: ModelServer) -> Build:
    follow_up_questions = REQUIRED_ARGUMENTS.get(server_backend)
    args = {}
    if follow_up_questions:
        for q, opts in follow_up_questions.items():
            args[q] = inquirer.text(
                message=q, completer=dict({o: None for o in opts})
            ).execute()

    return Build(model_server=server_backend, arguments=args)


def ask_name() -> str:
    return inquirer.text(message="What's the name of your model?").execute()
