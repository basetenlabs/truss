# Instance Groups Feature for Truss

This document describes the new instance groups feature that provides NVIDIA Triton-like multiple instance group support in Truss.

## Overview

Instance groups allow you to run multiple copies of your model concurrently, with automatic round-robin request distribution between them. This is particularly useful for models that are not thread-safe or where adding batching support would be complex.

## Configuration

Add the `instance_groups` field to your `config.yaml` under the `runtime` section:

```yaml
runtime:
  # Per-instance concurrency limit (existing field)
  predict_concurrency: 1
  # NEW: Number of model instances to create 
  instance_groups: 4
```

### Configuration Fields

- `predict_concurrency` (int, default: 1): Maximum concurrent requests per individual instance
- `instance_groups` (int, default: 1): Number of model instances to create and run in parallel

### Example Configurations

#### Single Instance (Default - Backward Compatible)
```yaml
runtime:
  predict_concurrency: 2
  instance_groups: 1  # Default behavior
```
Result: One model instance handling up to 2 concurrent requests.

#### Multiple Instances (New Feature)
```yaml
runtime:
  predict_concurrency: 1  
  instance_groups: 4
```
Result: Four model instances, each handling up to 1 concurrent request. Total capacity = 4 concurrent requests.

#### High Concurrency
```yaml
runtime:
  predict_concurrency: 2
  instance_groups: 4
```
Result: Four model instances, each handling up to 2 concurrent requests. Total capacity = 8 concurrent requests.

## Architecture

### How It Works

1. **Instance Pool Creation**: During model loading, the framework creates N separate instances of your Model class
2. **Round-Robin Distribution**: Incoming requests are distributed round-robin between available instances
3. **Per-Instance Concurrency**: Each instance has its own semaphore controlling `predict_concurrency`
4. **Health Management**: Unhealthy instances are automatically removed from the rotation

### Memory and Resource Considerations

- Each instance loads the model independently, so memory usage scales with `instance_groups`
- Consider your model size and available memory when setting `instance_groups`
- CPU usage also scales - ensure sufficient CPU resources are allocated

### Thread Safety Benefits

- Each instance has its own state, eliminating race conditions
- No need for complex thread synchronization in your model code
- Simplifies development of non-thread-safe models

## Migration Guide

### From Single Instance to Multiple Instances

**Before (config.yaml):**
```yaml
runtime:
  predict_concurrency: 4
  instance_groups: 1  # Default
```

**After (config.yaml):**
```yaml
runtime:
  predict_concurrency: 1
  instance_groups: 4  # Multiple instances
```

**Key Changes:**
- No changes needed to your model.py code
- Same total throughput (4 concurrent requests)
- Better thread safety with separate instances

## Monitoring and Observability

### Request Distribution
- Requests are distributed round-robin between healthy instances
- Failed/unhealthy instances are automatically skipped
- Tracing spans include instance information for debugging

### Metrics
- Each instance contributes to overall request metrics
- Individual instance health is monitored
- Total throughput = `instance_groups * predict_concurrency`

## Use Cases

### 1. Thread-Safe Models Without Batching
```python
class Model:
    def __init__(self):
        self.model = load_large_model()  # Thread-unsafe loading
    
    def predict(self, inputs):
        # No need for thread synchronization
        return self.model.process(inputs)
```

### 2. Legacy Models Not Designed for Concurrency
```python
class LegacyModel:
    def predict(self, inputs):
        # Uses global state, not thread-safe
        global_state.timestamp = time.time()
        return legacy_process(inputs)
```

### 3. High Throughput Requirements
```yaml
runtime:
  predict_concurrency: 1
  instance_groups: 8  # 8x parallel processing
```

## Limitations

1. **Memory Usage**: Each instance loads the full model independently
2. **Initialization Time**: Multiple instances increase startup time
3. **Cold Start**: All instances are loaded during startup, not on-demand

## Implementation Details

### Classes Added

1. **ModelInstance**: Represents a single model instance with its own semaphore
2. **ModelInstancePool**: Manages multiple instances and round-robin distribution

### Key Modifications

1. **TrussConfig**: Added `instance_groups` field with validation
2. **ModelWrapper**: Integrated instance pool into request processing
3. **Request Handling**: Modified to use per-instance semaphores

### Backward Compatibility

- Default `instance_groups: 1` preserves existing behavior
- All existing APIs remain unchanged
- Single instance path uses existing code paths

## Testing

The feature includes comprehensive tests for:
- Round-robin request distribution
- Backward compatibility with single instance
- Configuration validation
- Health management

Run tests with:
```bash
python test_instance_groups_basic.py
```
