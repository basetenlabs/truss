# BEIv3 encoder serving

The top-level `bei:` configuration serves encoder checkpoints with BEIv3.
Checkpoints load directly in the serving process; there is no TensorRT engine
compilation job and no `model.py` is required. The existing `trt_llm:` BEI and
Briton configurations continue to work for existing deployments.

Truss derives `docker_server.no_build: true` and a command that runs the native
`/usr/local/bin/bei` executable. It adds no Python launcher, virtual environment,
Nginx, or supervisord to the BEIv3 image.

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

The derived configuration declares `/v1/embeddings` as the predict endpoint and
`/health` for readiness and liveness, with BEIv3 listening on port 7997. The
platform must forward these routes directly; Truss installs no in-image proxy.

## Image selection

BEIv3 uses the existing `ImageVersions` payload and
`--trt_llm_image_versions_json` injection argument used by BEI, BEI-BERT,
Briton, and inference stack v2. The only additional payload field is
`bei_v3_image`, a fully resolved reference for the selected backend:

```json
{
  "bei_image": "registry/bei:legacy",
  "beibert_image": "registry/beibert:legacy",
  "briton_image": "registry/briton:legacy",
  "v2_llm_image": "registry/v2:approved",
  "bei_v3_image": "registry/bei:approved-nvidia"
}
```

Extend the existing platform resolver to select the BEIv3 image using `bei.backend` and
`bei.version_override`, defaulting to the approved version when the override is
null. Truss consumes `bei_v3_image` as supplied, preserving registry paths and
digests. There is no separate BEIv3 injection argument or payload type. Legacy
payloads without `bei_v3_image` remain valid for legacy engines.

The image must be resolved before no-build dispatch when context preparation is
skipped. The same shared argument remains available for local/fallback build
context preparation. No default image is hardcoded or published by this feature.

The platform rollout also needs to honor the derived `start_command` as a shell
command overriding the image's default command, forward the declared predict
endpoint for no-build deployments, and inject the declared environment variables
and secret mounts. In particular, the TPU image's default `--help` command must
not run. Those platform changes are separate from this Truss integration.

For local testing, an explicit `base_image.image` can supply a compatible BEIv3
image. `version_override` is forwarded to the platform resolver, like legacy
engine overrides; Truss does not construct image references from it. An injected
resolved image is authoritative. Without injection, local preparation requires
an explicit image and fails rather than selecting a generic Truss/CUDA image.

TPU deployments also require the TPU resource definitions and platform device,
topology, and socket setup. This Truss change prepares the serving image; it does
not install those host facilities. Use `resources.accelerator` when the TPU
accelerator types are available, or an explicit platform `instance_type`.

## Credentials and tuning

For private or gated Hugging Face checkpoints, declare `hf_access_token` in the
standard top-level `secrets` list. The native start command reads the mounted
secret or `TRUSS_SECRET_hf_access_token` at runtime and passes it as `BEI_HF_TOKEN`. An
existing `BEI_HF_TOKEN` or `HF_TOKEN` environment value also works. Tokens are not
inserted into the generated command. Runtime-mounted values are not written into
generated files. Explicit secret defaults supplied in the config retain standard Truss semantics;
use the secret manager for deployment credentials.

Advanced router settings remain available through top-level
`environment_variables` using the image's `BEI_*` settings. For TPU, Truss
defaults `BEI_MAX_BATCH_REQUESTS` to `32` when it is unset. Other batch and sequence
limits retain the image/runtime defaults. Adjust startup and restart checks for
the selected backend; TPU model warmup can take tens of minutes.

`bei:` manages its Docker-server command and routes. It cannot be combined with
`trt_llm:`, `bis_llm:`, custom Docker-server settings, live reload, or a non-HTTP
transport. With `no_build`, package installation, build commands, external data
downloads, and local package copying cannot run. Include dependencies and files
in the prebuilt image or use runtime weights mounts. The NVIDIA BEIv3 image stays
Python-free; the TPU image retains its existing Python backend dependencies.
This addition does not remove or migrate existing BEI/Briton models.
