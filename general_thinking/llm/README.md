---
library_id: gpt-oss-20b-latency
display_name: GPT-OSS 20B Latency
public: true
---

# GPT OSS 20B with BISv2 — High-Throughput Template

GPT OSS 20B is OpenAI's open source model designed for powerful reasoning, agentic tasks and other developer use cases. It uses their open source response format, Harmony.

This directory contains a **[Truss](https://truss.baseten.co/)** template for deploying **GPT OSS 20B** with **Baseten Inference Stack v2 (TensorRT-LLM + PyTorch backend)** on 4 H100 GPUs. This truss fully abstracts OpenAI's harmony response format, so everything works outside of the box. You can simply use it like a regular OpenAI compatible server. This stack maximizes both inference and throughput.

---

# Requirements

`truss==0.10.5`

You also need the file in data, which downloads GPT's harmony encoding ahead of time, because once deployed, the deployment will be unable to download from internet.

The environment variable `TIKTOKEN_RS_CACHE_DIR: /app/data` in `config.yaml` points `openai_harmony` to the local encoding file. See this(discussion)[https://huggingface.co/openai/gpt-oss-20b/discussions/39] for details.

---

## Core TRT-LLM `runtime` parameters

| Property (YAML path)       | Value                | Why it matters                                                             |
| -------------------------- | -------------------- | -------------------------------------------------------------------------- |
| `tensor_parallel_size`     | **4**                | Shards every weight matrix across the 2 H100s                              |
| `moe_expert_parallel_size` | **4**                | Shards each expert across 2 H100s                                          |
| `max_batch_size`           | **64**               | Up to 64 concurrent requests per forward pass                              |
| `max_seq_len`              | **98304**            | 98304 token context length                                                 |
| `enable_chunked_prefill`   | `true`               | Chunks long prompts to reduce memory usage                                 |
| `max_num_tokens`           | **8192**             | Upper limit on total tokens per chunk                                      |
| `served_model_name`        | `openai/gpt-oss-20b` | `model: openai/gpt-oss-20b` to call this model in OpenAI Compatible server |

---

## Important Advanced **`runtime.patch_kwargs`** parameters

These map 1-to-1 to TensorRT-LLM flags for extra performance tuning.

| Property (YAML path)                       | Value / Setting | Effect                                                    |
| ------------------------------------------ | --------------- | --------------------------------------------------------- |
| `cuda_graph_config.enable_padding`         | `true`          | Pad to fixed shape so one CUDA Graph is reused every step |
| `kv_cache_config.free_gpu_memory_fraction` | **0.8**         | 80 % of post-load VRAM reserved for paged KV-cache        |
| `kv_cache_config.enable_block_reuse`       | `true`          | Identical prefixes share cache blocks → faster TTFT       |
| `kv_cache_config.enable_block_reuse`       | `true`          | Identical prefixes share cache blocks → faster TTFT       |
| `chat_processor`                           | `harmony`       | GPT OSS uses Harmony response format                      |

---

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd openai/gpt-oss-20b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `openai/gpt-oss-20b` as your working directory, you can deploy the model with:

```sh
truss push --trusted
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with PyTorch Backend will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
