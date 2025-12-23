# Weights Configuration Design

**Date:** December 2024  
**Status:** Approved

## Summary

The `weights` key in Truss config provides a cleaner alternative to `model_cache` for specifying model weights from multiple sources (HuggingFace, S3, GCS, Azure). It uses a URI-based `source` field that eliminates redundancy and improves clarity.

## Design Decisions

### 1. Use `source` with URI prefixes instead of `repo_id` + `kind`

**Problem:** The current `model_cache` has redundancy - you specify both `kind: s3` AND `repo_id: "s3://bucket/path"`. The `kind` field is only truly needed for HuggingFace because HF repo IDs don't have a URI scheme.

**Solution:** Use a single `source` field with URI prefixes. The backend infers the source type from the scheme:

```yaml
weights:
  # HuggingFace - no prefix (default) or explicit hf://
  - source: "meta-llama/Llama-2-7b"
    revision: "main"
    mount_location: "/models/llama"
    
  # S3
  - source: "s3://my-bucket/models/llama"
    mount_location: "/models/llama"
    
  # GCS  
  - source: "gs://my-bucket/models/llama"
    mount_location: "/models/llama"
    
  # Azure
  - source: "azure://account/container/path"
    mount_location: "/models/llama"
```

**Rationale:** 
- Eliminates `kind` field entirely - less redundancy
- URI format is familiar (S3/GCS already use this convention)
- If no scheme prefix, assume HuggingFace (most common case, backward-compatible mental model)

### 2. No `use_volume` field

**Problem:** `model_cache` has `use_volume: bool` which controls whether weights are mounted via CSI volume or downloaded at build time.

**Solution:** The `weights` key always implies volume-based loading. No `use_volume` field needed.

**Rationale:** The whole point of the new `weights` key is to use the improved volume-based approach. Users who want the old behavior can continue using `model_cache`.

### 3. Rename `volume_folder` to `mount_location`

**Problem:** `volume_folder` is ambiguous - it sounds like a folder name, but it's actually the full mount path.

**Solution:** Rename to `mount_location` and validate that it's an absolute path (starts with `/`).

```yaml
mount_location: "/models/llama"  # Must be absolute path
```

### 4. `revision` only valid for HuggingFace

**Problem:** The `revision` field (git ref/commit SHA) only makes sense for HuggingFace. For S3/GCS/Azure, there's no concept of revision.

**Solution:** Error if `revision` is set when source is not HuggingFace.

```yaml
# Valid - HF with revision
- source: "meta-llama/Llama-2-7b"
  revision: "main"

# Invalid - S3 doesn't support revision
- source: "s3://bucket/path"
  revision: "main"  # ERROR
```

### 5. Keep `runtime_secret_name` unchanged

**Problem:** Should we change how secrets work?

**Solution:** Keep the existing `runtime_secret_name` pattern. It already works well:
- Single secret name references a file containing credentials
- For multi-credential sources (S3, Azure), the file contains JSON with all credentials
- Aligns with industry patterns (AWS profiles, Terraform providers, rclone)

```yaml
runtime_secret_name: "aws_credentials"  # JSON: {"access_key_id": "...", "secret_access_key": "...", "region": "..."}
```

## Final Schema

```yaml
weights:
  - source: str              # Required. URI with scheme (s3://, gs://, azure://) or HF repo ID
    revision: str            # Optional. Only valid for HuggingFace sources
    mount_location: str      # Required. Absolute path where weights are mounted
    runtime_secret_name: str # Optional. Secret containing credentials
    allow_patterns: list     # Optional. File patterns to include
    ignore_patterns: list    # Optional. File patterns to exclude
```

## Examples

### HuggingFace

```yaml
weights:
  - source: "meta-llama/Llama-2-7b"
    revision: "main"
    mount_location: "/models/llama"
    runtime_secret_name: "hf_access_token"
    allow_patterns:
      - "*.safetensors"
      - "config.json"
```

### S3

```yaml
weights:
  - source: "s3://my-bucket/models/llama-7b"
    mount_location: "/models/llama"
    runtime_secret_name: "aws_credentials"
```

### GCS

```yaml
weights:
  - source: "gs://my-bucket/models/llama-7b"
    mount_location: "/models/llama"
    runtime_secret_name: "gcp_service_account"
```

### Azure

```yaml
weights:
  - source: "azure://myaccount/container/llama-7b"
    mount_location: "/models/llama"
    runtime_secret_name: "azure_credentials"
```

### Multi-source

```yaml
weights:
  - source: "meta-llama/Llama-2-7b"
    revision: "main"
    mount_location: "/models/base"
    runtime_secret_name: "hf_access_token"
    
  - source: "s3://my-lora-adapters/custom-v1"
    mount_location: "/models/adapter"
    runtime_secret_name: "aws_credentials"
```

## Migration

- `model_cache` remains unchanged for backward compatibility
- Users cannot specify both `model_cache` and `weights` (existing validation)
- New deployments should prefer `weights` over `model_cache`


