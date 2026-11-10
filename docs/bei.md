# BEIv3 encoder serving

The top-level `bei:` configuration serves encoder checkpoints with BEIv3.
Checkpoints load directly in the serving process; there is no TensorRT engine
compilation job and no `model.py` is required. The existing `trt_llm:` BEI and
Briton configurations continue to work for existing deployments.

```yaml
model_name: Embedding encoders
resources:
  accelerator: L4
bei:
  backend: nvidia
  version_override: null
  configuration:
    - checkpoint: BAAI/bge-small-en-v1.5
      revision: main
      served_model_name: baai-small
      quantization: none
    - checkpoint: BAAI/bge-large-en-v1.5
      revision: main
      served_model_name: baai-large
      quantization: fp8
```

`backend` selects the prebuilt `nvidia` or `tpu` BEIv3 image. Each entry in
`configuration` specifies a Hugging Face checkpoint or an absolute path to a
checkpoint already present in the container, an optional revision, an optional
public model name, and a quantization mode. Revisions default to `main`; use an
immutable revision for reproducible deployments. Public names default to the
checkpoint and must be unique within the deployment.

The initial implementation supports `none` and NVIDIA dynamic `fp8` quantization.
FP8 requires SM89 or newer hardware and a compatible, unquantized checkpoint.
MXFP8 and NVFP4 need runtime support and are rejected by this version. The TPU
backend currently accepts one model with `quantization: none` and uses BF16
compute. Its image supplies the TorchTPU plugin configuration.

On NVIDIA, a single model uses BEIv3's default replication across visible GPUs.
Multiple models also replicate across all visible GPUs. This allows small
models to share a GPU; their combined weights and runtime buffers must fit on
each GPU. This version supports one node per deployment.

Requests use the OpenAI embeddings shape. Include the served model name when
multiple models are deployed:

```json
{"model": "baai-small", "input": ["Text to embed"]}
```

The standard Baseten predict route maps to `/v1/embeddings`. BEIv3 runs on
internal port 7997, with `/health` used for readiness and liveness. Normal
Docker-server proxying forwards its other routes too.

## Image selection

The backend must supply its approved BEIv3 image references during build setup:

```sh
truss-docker-build-setup --truss_type server \
  --bei_image_versions_json '{"nvidia_image":"registry/bei:approved-nvidia","tpu_image":"registry/bei:approved-tpu"}'
```

This is the Truss side of the injection contract; platform image resolution and
rollout must pass this new argument. It is independent of the legacy
`--trt_llm_image_versions_json` argument. Default image tags are not hardcoded in
Truss, and no image is published by this feature.

For testing, an explicit `base_image.image` can supply a compatible BEIv3 image.
Alternatively, `version_override: v3.0.1-<commit>` selects
`baseten/bei:v3.0.1-<commit>-nvidia` or `-tpu` according to `backend`. An override
takes precedence over an explicit base image; an explicit base image takes
precedence over an injected default. An unresolved image fails before generating
a fallback Truss/CUDA image. These tags illustrate the existing BEIv3 naming
scheme; use an image that has actually been published.

TPU deployments also require the TPU resource definitions and platform device,
topology, and socket setup. This Truss change prepares the serving image; it does
not install those host facilities. Use `resources.accelerator` when the TPU
accelerator types are available, or an explicit platform `instance_type`.

## Credentials and tuning

For private or gated Hugging Face checkpoints, declare `hf_access_token` in the
standard top-level `secrets` list. The launcher reads the mounted secret or
`TRUSS_SECRET_hf_access_token` at runtime and passes it as `BEI_HF_TOKEN`. An
existing `BEI_HF_TOKEN` or `HF_TOKEN` environment value also works. Tokens are not
inserted into the generated command. Runtime-mounted values are not written into
launcher settings.
Explicit secret defaults supplied in the config retain standard Truss semantics;
use the secret manager for deployment credentials.

Advanced router settings remain available through top-level
`environment_variables` using the image's `BEI_*` settings. For TPU, the launcher
defaults `BEI_MAX_BATCH_REQUESTS` to `32` when it is unset. Other batch and sequence
limits retain the image/runtime defaults. Adjust startup and restart checks for
the selected backend; TPU model warmup can take tens of minutes.

`bei:` manages its Docker-server command and routes. It cannot be combined with
`trt_llm:`, `bis_llm:`, custom Docker-server settings, live reload, or a non-HTTP
transport. Include custom Python dependencies in the BEIv3 base image rather than
top-level `requirements`. The native NVIDIA base image does not need a Python
installation: Truss supplies a managed interpreter for its launcher/proxy.
This addition does not remove or migrate existing BEI/Briton models.
