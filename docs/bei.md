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

The backend must resolve its approved BEIv3 image references before dispatching
the no-build deployment. The image-version payload has this shape:

```json
{"nvidia_image":"registry/bei:approved-nvidia","tpu_image":"registry/bei:approved-tpu"}
```

The build-context CLI accepts this payload through `--bei_image_versions_json`
for local/fallback preparation. A deployment that skips that step must resolve
the image earlier. Default image tags are not hardcoded in Truss, and no image
is published by this feature.

The platform rollout also needs to honor the derived `start_command` as a shell
command overriding the image's default command, forward the declared predict
endpoint for no-build deployments, and inject the declared environment variables
and secret mounts. In particular, the TPU image's default `--help` command must
not run. Those platform changes are separate from this Truss integration.

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
