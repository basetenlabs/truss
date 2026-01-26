# Truss-Transfer

Python-optional download utility for resolving Baseten Pointers (bptr).

## Installation

```bash
pip install truss-transfer
# pip install /workspace/model-performance/michaelfeil/truss/truss-transfer/target/wheels/truss_transfer-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```

## How to Resolve a bptr

### Via Python Package

```python
import truss_transfer

# Resolve bptr using default download directory from environment
result_dir = truss_transfer.lazy_data_resolve()

# Resolve bptr with custom download directory
result_dir = truss_transfer.lazy_data_resolve("/custom/download/path")

# Example usage in a data loader
def lazy_data_loader(download_dir: str):
    print(f"download using {truss_transfer.__version__}")
    try:
        resolved_dir = truss_transfer.lazy_data_resolve(str(download_dir))
        print(f"Files resolved to: {resolved_dir}")
        return resolved_dir
    except Exception as e:
        print(f"Lazy data resolution failed: {e}")
        raise
```

### Via CLI

```bash
# Using the compiled binary
./target/x86_64-unknown-linux-musl/release/truss_transfer_cli /tmp/download_dir

# Using the Python package CLI
python -m truss_transfer /tmp/download_dir
```

## How to Build a bptr and Save it via Python

You can create Baseten Pointers from HuggingFace models using the Python API:

```python
import truss_transfer
import json

# Define models to include in the bptr
models = [
    truss_transfer.PyModelRepo(
        repo_id="microsoft/DialoGPT-medium",
        revision="main",
        volume_folder="dialogpt",
        kind="hf",  # "hf" for HuggingFace, "gcs" for Google Cloud Storage
        runtime_secret_name="hf_access_token",
        allow_patterns=["*.safetensors", "*.json"],  # Optional: specific file patterns
        ignore_patterns=["*.txt"]  # Optional: patterns to ignore
    ),
    truss_transfer.PyModelRepo(
        repo_id="julien-c/dummy-unknown",
        revision="60b8d3fe22aebb024b573f1cca224db3126d10f3",
        volume_folder="julien_dummy",
        runtime_secret_name="hf_access_token_2"
    )
]

# Create the bptr manifest
bptr_manifest = truss_transfer.create_basetenpointer_from_models(models)

# Save to file
with open("/bptr/static-bptr-manifest.json", "w") as f:
    f.write(bptr_manifest)

# Or parse as JSON for programmatic use
manifest_data = json.loads(bptr_manifest)
print(f"Created bptr with {len(manifest_data)} pointers")
```

### PyModelRepo Parameters

- **`repo_id`**: Repository identifier (e.g., "microsoft/DialoGPT-medium")
- **`revision`**: Git commit hash or branch name (e.g., "main", commit hash)
- **`volume_folder`**: Local folder name where files will be stored
- **`kind`**: Repository type - "hf" for HuggingFace, "gcs" for Google Cloud Storage
- **`runtime_secret_name`**: Name of the secret containing access token
- **`allow_patterns`**: Optional list of file patterns to include
- **`ignore_patterns`**: Optional list of file patterns to exclude

## End-to-End Flow

Here's a complete example of creating and resolving a bptr:

### Step 1: Create a bptr Manifest

```python
import truss_transfer
import json
import os

# Create models configuration
models = [
    truss_transfer.PyModelRepo(
        repo_id="microsoft/DialoGPT-medium",
        revision="main",
        volume_folder="dialogpt",
        runtime_secret_name="hf_access_token"
    ),
    truss_transfer.PyModelRepo(
        repo_id="gs://llama-3-2-1b-instruct/",
        revision="",
        volume_folder="llama",
        # requires json in /secrets/gcs-service-account-jsn
        runtime_secret_name="gcs-service-account-jsn",
        kind="gcs"
    ),
    truss_transfer.PyModelRepo(
        repo_id="s3://bt-training-dev-org-b68c04fe47d34c85bfa91515bc9d5e2d/training_projects",
        revision="",
        volume_folder="training",
        # requires json in /secrets/aws
        runtime_secret_name="aws-secret-json",
        kind="s3"
    )
]

# Generate the bptr manifest
bptr_manifest = truss_transfer.create_basetenpointer_from_models(models)

# Ensure the directory exists
os.makedirs("/bptr", exist_ok=True)

# Save the manifest
with open("/bptr/static-bptr-manifest.json", "w") as f:
    f.write(bptr_manifest)

print("bptr manifest created successfully!")
```

### Step 2: Set up Environment (Optional)

```bash
# Configure download location
export TRUSS_TRANSFER_DOWNLOAD_DIR="/tmp/my-models"

# Enable b10fs caching (optional)
export BASETEN_FS_ENABLED=1

# Set up authentication (if needed)
export HF_TOKEN="your-huggingface-token"
# Or use the official HuggingFace environment variable
export HUGGING_FACE_HUB_TOKEN="your-huggingface-token"
```

### Step 3: Resolve the bptr

```python
import truss_transfer

# Resolve the bptr - downloads files to the specified directory
resolved_dir = truss_transfer.lazy_data_resolve("/tmp/my-models")
print(f"Files downloaded to: {resolved_dir}")

# Now you can use the downloaded files
import os
files = os.listdir(resolved_dir)
print(f"Downloaded files: {files}")
```

### Step 4: Use the Downloaded Files

```python
# Example: Load a model from the resolved directory
model_path = os.path.join(resolved_dir, "dialogpt")
# Your model loading code here...
```

### Complete Workflow

```python
# Complete example combining creation and resolution
import truss_transfer
import json
import os

def create_and_resolve_bptr():
    # runtime_secret_name: best to be created with `-` in baseten.
    # 1. Create bptr manifest
    models = [
        truss_transfer.PyModelRepo(
            repo_id="NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4",
            revision="main",
            # write to folder named
            volume_folder="dialogpt",
            # read secret from /secrets/hf-access-token
            runtime_secret_name="hf-access-token"
        ),
        # requires a gcs service account json
    ]
    root = "/tmp/my-models"
    bptr_manifest = truss_transfer.create_basetenpointer_from_models(models, root)

    # 2. Save manifest
    os.makedirs("/static-bptr", exist_ok=True)
    with open("/static-bptr/static-bptr-manifest.json", "w") as f:
        f.write(bptr_manifest)

    # 3. Resolve bptr. If we would set `root` above to "", we could define the base dir here.
    truss_transfer.lazy_data_resolve(root)

    # 4. Verify files were downloaded
    dialogpt_path = os.path.join(root, "dialogpt")
    if os.path.exists(dialogpt_path):
        files = os.listdir(dialogpt_path)
        print(f"Successfully downloaded {len(files)} files to {dialogpt_path}")
        return dialogpt_path
    else:
        raise Exception("Model files not found after resolution")

# Run the workflow
model_path = create_and_resolve_bptr()
```

### Secrets
Preferably, use a `-` to and lowercase characters to add credentials in baseten.

#### AWS
```json
{
  "access_key_id": "XXXXX",
  "secret_access_key": "adada/adsdad",
  "region": "us-west-2"
}
```

#### Google GCS
```json
{
      "private_key_id": "b717a4db1dd5a5d1f980aef7ea50616584b6ebc8",
      "private_key": "-----BEGIN PRIVATE KEY-----\nMI",
      "client_email": "b10-some@xxx-example.iam.gserviceaccount.com"
}
```

#### Huggingface
The Huggingface token.

#### Azure
(Untested)
```json
{
    "account_key": "key",
}
```
## Environment Variables and Settings

The following environment variables can be used to configure truss-transfer behavior:

### Core Configuration

- **`TRUSS_TRANSFER_DOWNLOAD_DIR`** (default: `/tmp/truss_transfer`)
  - Directory where resolved files will be downloaded
  - Used when no explicit download directory is provided
  - Can be overridden by passing a directory to the CLI or Python function

- **`TRUSS_TRANSFER_LOG`** or **`RUST_LOG`** (default: `info`)
  - Controls logging level: `error`, `warn`, `info`, `debug`, `trace`
  - `TRUSS_TRANSFER_LOG` takes precedence over `RUST_LOG`
  - Example: `RUST_LOG=debug` for detailed logging

- **`TRUSS_TRANSFER_CACHE_DIR`** (default: `/cache/org/artifacts/truss_transfer_managed_v1`)
  - Cache directory for b10fs operations
  - Used when Baseten FS is enabled

### Download Configuration

- **`TRUSS_TRANSFER_NUM_WORKERS`** (default: `6`)
  - Number of concurrent download workers
  - Controls parallelism for file downloads

- **`TRUSS_TRANSFER_USE_RANGE_DOWNLOAD`** (default: `true`)
  - Enable/disable range-based downloading for large files
  - Set to `1`, `true`, `yes`, or `y` to enable

- **`TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS`** (default: `192`)
  - Total number of range download workers across all files
  - Used when range downloading is enabled

- **`TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS_PER_FILE`** (default: `84`)
  - Number of concurrent range workers per individual file
  - Used when range downloading is enabled

- **`TRUSS_TRANSFER_DOWNLOAD_MONITOR_SECS`** (default: `30`)
  - Interval in seconds for monitoring download progress
  - Controls how often progress is reported

- **`TRUSS_TRANSFER_PAGE_AFTER_DOWNLOAD`** (default: `false`)
  - Enable/disable memory paging after downloads complete
  - Set to `1`, `true`, `yes`, or `y` to enable
  - Helps with memory management for large downloads

### Authentication

- **`HF_TOKEN`** (optional)
  - HuggingFace access token for accessing private repositories
  - Takes precedence over `HUGGING_FACE_HUB_TOKEN`
  - Used when `runtime_secret_name` is `hf_token` or `hf_access_token`

- **`HUGGING_FACE_HUB_TOKEN`** (optional)
  - Official HuggingFace Hub token environment variable
  - Used as fallback if `HF_TOKEN` is not set
  - Allows access to private HuggingFace repositories

### Baseten FS (b10fs) Configuration

- **`BASETEN_FS_ENABLED`** (default: `false`)
  - Enable/disable Baseten FS caching: `1`/`true` to enable, `0`/`false` to disable
  - When enabled, files are cached in the directory specified by `TRUSS_TRANSFER_CACHE_DIR`

- **`TRUSS_TRANSFER_B10FS_CLEANUP_HOURS`** (default: `96`)
  - Hours after last access before deleting cached files from other tenants
  - Helps manage disk space by removing old cached files
  - Example: `TRUSS_TRANSFER_B10FS_CLEANUP_HOURS=48` for 2 days

- **`TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS`** (default: dynamic)
  - Expected download speed in MB/s for b10fs performance benchmarking
  - Used to determine if b10fs is faster than direct download
  - Default: 400 MB/s for >16 cores, 90 MB/s for ≤16 cores (with randomization)
  - Lower values make b10fs more likely to be used

- **`TRUSS_TRANSFER_B10FS_MAX_STALE_CACHE_SIZE_GB`** (default: unlimited)
  - Maximum size in GB for stale cache files before cleanup is triggered
  - When set, actively purges old cache files to maintain this limit
  - Example: `TRUSS_TRANSFER_B10FS_MAX_STALE_CACHE_SIZE_GB=500`

### Example Configuration

```bash
# Basic setup
export TRUSS_TRANSFER_DOWNLOAD_DIR="/tmp/my-models"
export TRUSS_TRANSFER_LOG=info
export TRUSS_TRANSFER_NUM_WORKERS=8

# Advanced download configuration
export TRUSS_TRANSFER_USE_RANGE_DOWNLOAD=1
export TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS=256
export TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS_PER_FILE=64
export TRUSS_TRANSFER_PAGE_AFTER_DOWNLOAD=1

# With b10fs enabled and tuned
export BASETEN_FS_ENABLED=1
export TRUSS_TRANSFER_CACHE_DIR="/fast-ssd/cache"
export TRUSS_TRANSFER_B10FS_CLEANUP_HOURS=48
export TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS=200
export TRUSS_TRANSFER_B10FS_MAX_STALE_CACHE_SIZE_GB=1000

# Authentication
export HF_TOKEN="your-huggingface-token"
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run tests without network dependencies
cargo test --lib

# Run Python tests
python -m pytest tests/
```

### Running the CLI as binary

Compiling the libary as musl-linux target for cross-platform usage.
```
# Add one-time installations
# apt-get install -y musl-tools libssl-dev libatomic-ops-dev
# rustup target add x86_64-unknown-linux-musl

# To build with cargo:
cargo build --release --target x86_64-unknown-linux-musl --features cli --bin truss_transfer_cli
```

```
# To run the binary
./target/x86_64-unknown-linux-musl/release/truss_transfer_cli /tmp/ptr
```

### Building a wheel from source

Prerequisites:
```sh
# apt-get install patchelf
# Install rust via Rustup https://www.rust-lang.org/tools/install
pip install maturin==1.8.1
```

This will build you the wheels for your current `python3 --version`.
The output should look like this:
```
maturin build --release
🔗 Found pyo3 bindings
🐍 Found CPython 3.9 at /workspace/model-performance/michaelfeil/.asdf/installs/python/3.9.21/bin/python3
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.75s
🖨  Copied external shared libraries to package truss_transfer.libs directory:
    /usr/lib/x86_64-linux-gnu/libssl.so.3
    /usr/lib/x86_64-linux-gnu/libcrypto.so.3
📦 Built wheel for CPython 3.9 to /workspace/model-performance/michaelfeil/truss/truss-transfer/target/wheels/truss_transfer-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```

### Release a new version and make it the default version used in the serving image builder for new deploys
truss-transfer gets bundled with truss in the context-builder phase. In this phase, the truss-transfer version gets installed.
To make truss-transfer bundeable, it needs to be published to pypi and github releases.

1. Open a PR with rust changes
2. Change the version to x.z.y+1.rc0 in Cargo.toml and push change to branch a.
3. Run a `Buid and Release truss-transfer" action https://github.com/basetenlabs/truss/actions with "release to pypi = true" on this branch a.
4. Make x.z.y+1.rc0  as truss pyproject.toml, and templates/server/requirements.txt dependency
5. Edit truss to a new truss.rcX, publish truss.rcX to pypy.org (main.yml action)
6. pip install truss=truss.rcX locally and truss push (on example that uses python truss)
7. Merge PR
8. Wait for CLI binary to be released under assets as part of a new tag (https://github.com/basetenlabs/truss/releases)
9. add the CLI to the server.Dockerfile.jinja to have it available for trussless.
