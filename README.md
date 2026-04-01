# Truss

**The simplest way to serve AI/ML models in production**

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/py/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

Truss is the CLI for deploying and serving ML models on Baseten. Package your model's serving logic in Python, launch training jobs, and deploy to production—Truss handles containerization, dependency management, and GPU configuration.

Truss lets you serve models with the [Baseten Inference Stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/) as well as deploy models from any open-source framework: vLLM, SGLang, TensorRT-LLM, `transformers`, `diffusers`, PyTorch, TensorFlow, and more.

**[Get started](https://docs.baseten.co/examples/deploy-your-first-model)** | [100+ examples](https://github.com/basetenlabs/truss-examples/) | [Documentation](https://docs.baseten.co)

# Why Truss?

* **Write once, run anywhere:** Package model code, weights, and dependencies with a model server that behaves the same in development and production.
* **Fast developer loop:** Iterate with live reload, skip Docker and Kubernetes configuration, and use a batteries-included serving environment.
* **Support for all Python frameworks:** From `transformers` and `diffusers` to PyTorch and TensorFlow to vLLM, SGLang, and TensorRT-LLM, Truss supports models created and served with any framework.
* **Production-ready:** Built-in support for GPUs, secrets, caching, and autoscaling when deployed to [Baseten](https://baseten.co) or your own infrastructure.

# Installation

Install Truss with:

```
pip install --upgrade truss
```

# IDE support

Truss ships a [JSON schema](truss/config.schema.json) for `config.yaml`. Projects created with `truss init` include a schema reference automatically, giving you autocompletion, hover docs, and validation in any editor that supports the [YAML language server](https://github.com/redhat-developer/yaml-language-server) (VS Code, JetBrains, Neovim, and others).

To add schema support to an existing `config.yaml`, add this comment as the first line:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/basetenlabs/truss/main/truss/config.schema.json
```

# Quickstart

Deploying a model to Baseten via Truss turns a Hugging Face model into a production-ready API endpoint. You write a `config.yaml` that specifies the model, the hardware, and the engine, then `uvx truss push` builds a TensorRT-optimized container and deploys it. No Python code, no Dockerfile, no container management.

This guide walks through deploying [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), a small but capable LLM, from a config file to a production API. You'll set up Truss, write a config, deploy to Baseten, call the model's OpenAI-compatible endpoint, and promote to production.

## Set up your environment

Before you begin:

- [Sign up](https://app.baseten.co/signup) or [sign in](https://app.baseten.co/login) to Baseten.
- Install [uv](https://docs.astral.sh/uv/), a fast Python package manager. This guide uses `uvx` to run [Truss](https://pypi.org/project/truss/) commands without a separate install step.

### Authenticate with Baseten

Generate an API key from [Settings > API keys](https://app.baseten.co/settings/account/api_keys), then log in:

```sh
uvx truss login
```

Paste your API key when prompted:

```output
💻 Let's add a Baseten remote!
🤫 Quietly paste your API_KEY:
```

You can skip the interactive prompt by setting `BASETEN_API_KEY` as an environment variable:
```bash
export BASETEN_API_KEY="paste-your-api-key-here"
```

## Create a Truss project

Scaffold a new project:

```sh
uvx truss init qwen-2.5-3b && cd qwen-2.5-3b
```

When prompted, name the model `Qwen 2.5 3B`.

```output
? 📦 Name this model: Qwen 2.5 3B
Truss Qwen 2.5 3B was created in ~/qwen-2.5-3b
```

This creates a directory with a `config.yaml`, a `model/` directory, and supporting files. For engine-based deployments like this one, you only need `config.yaml`. The `model/` directory is for [custom Python code](/examples/customize-a-model) when you need custom preprocessing, postprocessing, or unsupported model architectures.

## Write the config

Replace the contents of `config.yaml` with:

```yaml config.yaml
model_name: Qwen-2.5-3B
resources:
  accelerator: L4
  use_gpu: true
trt_llm:
  build:
    base_model: decoder
    checkpoint_repository:
      source: HF
      repo: "Qwen/Qwen2.5-3B-Instruct"
    max_seq_len: 8192
    quantization_type: fp8
    tensor_parallel_count: 1
```

That's the entire deployment specification.

- `model_name` identifies the model in your Baseten dashboard.
- `resources` selects an L4 GPU (24 GB VRAM), which is plenty for a 3B parameter model.
- `trt_llm` tells Baseten to use [Engine-Builder-LLM](/engines/engine-builder-llm/overview), which compiles the model with TensorRT-LLM for optimized inference.
- `checkpoint_repository` points to the model weights on Hugging Face. Qwen 2.5 3B Instruct is ungated, so no access token is needed.
- `quantization_type: fp8` compresses weights to 8-bit floating point, cutting memory usage roughly in half with negligible quality loss.
- `max_seq_len: 8192` sets the maximum context length for requests.

---

## Deploy

Push the model to Baseten:

We'll start by deploying in development mode so we can iterate quickly:

```sh
uvx truss push --watch
```

You should see:

```output
✨ Model Qwen 2.5 3B was successfully pushed ✨

🪵  View logs for your deployment at https://app.baseten.co/models/abc1d2ef/logs/xyz123
👀 Watching for changes to truss...
```

The logs URL contains your model ID, the string after `/models/` (e.g., `abc1d2ef`). You'll need this to call the model's API. You can also find it in your [Baseten dashboard](https://app.baseten.co/models/).

Baseten now downloads the model weights from Hugging Face, compiles them with TensorRT-LLM, and deploys the resulting container to an L4 GPU. You can watch progress in the logs linked above.

## Call the model

Engine-based deployments serve an OpenAI-compatible API. Once the deployment shows "Active" in the dashboard, call it using the OpenAI SDK or cURL. Replace `{model_id}` with your model ID from the deployment output.

Install the OpenAI SDK if you don't have it:

```sh
uv pip install openai
```

Create a chat completion:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-{model_id}.api.baseten.co/environments/development/sync/v1",
)

response = client.chat.completions.create(
    model="Qwen-2.5-3B",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
)

print(response.choices[0].message.content)
```

You should see a response like:

```output
Machine learning is a branch of artificial intelligence where systems learn
patterns from data to make predictions or decisions without being explicitly
programmed for each task...
```

Any code that works with the OpenAI SDK works with your deployment. Just point the `base_url` at your model's endpoint.

## Iterate with live reload

When you change your `config.yaml` and want to test quickly, use live reload:

```sh
uvx truss watch
```

You should see:

```output
🪵  View logs for your deployment at https://app.baseten.co/models/<model_id>/logs/<deployment_id>
🚰 Attempting to sync truss with remote
No changes observed, skipping patching.
👀 Watching for changes to truss...
```

When you save changes, Truss automatically syncs them with the deployed model. This saves time by patching without a full rebuild.

If you stopped the watch session, you can re-attach with:

```sh
uvx truss watch
```

This creates a production deployment with its own endpoint. The API URL changes from `/environments/development/` to `/environments/production/`:

```python
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-{model_id}.api.baseten.co/environments/production/sync/v1",
)
```

Your model ID is the string after `/models/` in the logs URL from `uvx truss push`. You can also find it in your [Baseten dashboard](https://app.baseten.co/models/).
