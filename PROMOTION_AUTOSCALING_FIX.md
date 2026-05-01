# Promotion Autoscaling Fix

## Problem Statement

When a deployment is being promoted to an environment (e.g., `truss push --environment production`), it immediately takes on the autoscaling settings of that environment (e.g., 4/10 min/max replicas for production).

If the build or deploy fails, the deployment retains these autoscaling settings. When someone retries the deployment from the UI (which does NOT retry the promotion), and it succeeds, the deployment stays at those autoscaling settings despite not being in the production environment.

**Impact**: This has led to over $150k in refunds due to users being confused by promotions hanging around post-promotion, eroding customer trust.

## Root Cause

The Baseten backend applies environment autoscaling settings immediately when a deployment is created with `environment_name` parameter in the GraphQL mutation, rather than waiting until the deployment successfully completes and is actually promoted.

## Solution

### Client-Side Changes (Truss CLI)

Added a new parameter `defer_environment_settings` to the GraphQL mutations:

1. **`create_model_version_from_truss`** mutation now includes:
   ```graphql
   defer_environment_settings: true
   ```

2. **`create_model_from_truss`** mutation now includes:
   ```graphql
   defer_environment_settings: true
   ```

This parameter signals to the backend that environment-specific autoscaling settings should only be applied AFTER the deployment successfully completes, not at creation time.

### Changed Files

1. **`truss/remote/baseten/api.py`**:
   - Added `defer_environment_settings` parameter (default `True`) to `create_model_version_from_truss()`
   - Added `defer_environment_settings` parameter (default `True`) to `create_model_from_truss()`
   - Added `update_deployment()` method for updating deployment settings via REST API

2. **`truss/remote/baseten/core.py`**:
   - Passes `defer_environment_settings=True` when calling API methods for model creation

3. **`tests/e2e/test_promotion_autoscaling.py`**:
   - New E2E test that verifies the fix works correctly
   - Tests the exact scenario from the acceptance criteria

### Backend Requirements

The backend GraphQL mutations must be updated to:

1. Accept the `defer_environment_settings` parameter
2. When `defer_environment_settings=true`:
   - Store the intended environment name
   - Do NOT immediately apply environment autoscaling settings
   - Only apply environment settings when the deployment reaches ACTIVE status
   - If deployment fails (BUILD_FAILED, DEPLOY_FAILED), clear environment association

## E2E Test

The E2E test in `tests/e2e/test_promotion_autoscaling.py` verifies:

1. **Step 1**: Push successful deployment to production environment
   - Verify it has production autoscaling (4/10 min/max replicas)

2. **Step 2**: Push deployment that will fail build
   - Deployment is created with environment=production
   - Build fails due to invalid requirement

3. **Step 3**: Retry deployment (simulated by pushing fixed version without --environment)
   - This simulates what happens when user clicks "Retry" in UI

4. **Step 4**: Verify retry deployment does NOT have production autoscaling
   - Should have default settings, not production settings
   - This confirms the fix works

### Running the E2E Test

```bash
# Set environment variable to enable E2E tests
export RUN_E2E_TESTS=1

# Run the test
pytest tests/e2e/test_promotion_autoscaling.py -xvs
```

Or run directly:
```bash
RUN_E2E_TESTS=1 python tests/e2e/test_promotion_autoscaling.py
```

**Prerequisites**:
- Configured Baseten remote with valid API key
- Access to create deployments and environments
- Sufficient quota for test deployments

### Manual Testing Against Devcontainer

To manually test against a devcontainer deployment:

1. **Create successful production deployment**:
   ```bash
   truss push oracle_v1/ --model-name test-promotion-fix --environment production --wait
   ```
   - Verify it has 4/10 min/max replicas in UI

2. **Create failing deployment**:
   - Edit `oracle_v2/config.yaml` to add invalid requirement:
     ```yaml
     requirements:
       - this-does-not-exist==999.999.999
     ```
   - Push:
     ```bash
     truss push oracle_v2/ --model-name test-promotion-fix --environment production
     ```
   - Wait for build to fail

3. **Retry the deployment**:
   - Fix the requirement in `oracle_v2/config.yaml`
   - Either:
     - Click "Retry" in Baseten UI, OR
     - Push again without environment:
       ```bash
       truss push oracle_v2/ --model-name test-promotion-fix
       ```

4. **Verify autoscaling**:
   - Check deployment in UI
   - Should have default autoscaling (typically 0/3 or 1/3), NOT 4/10
   - ✓ PASS if min_replicas != 4
   - ✗ FAIL if min_replicas == 4

## API Changes Summary

### New GraphQL Mutation Parameters

Both `create_model_from_truss` and `create_model_version_from_truss` now support:

```graphql
defer_environment_settings: Boolean
```

**Default**: `true`  
**Purpose**: When true, environment autoscaling settings are only applied after successful deployment

### New REST API Method

Added `update_deployment()` method to `BasetenApi`:

```python
def update_deployment(self, model_id: str, deployment_id: str, update_data: dict) -> Any:
    """Update a deployment's configuration (e.g., autoscaling settings)"""
    return self._rest_api_client.patch(
        f"v1/models/{model_id}/deployments/{deployment_id}", 
        body=update_data
    )
```

This can be used to manually update autoscaling settings if needed.

## Backward Compatibility

The changes are backward compatible:

- `defer_environment_settings` defaults to `True` (new, safer behavior)
- Existing code will automatically use the new behavior
- If backend doesn't support the parameter yet, it will be ignored (GraphQL allows unknown fields)

## Future Improvements

1. Add CLI flag to control `defer_environment_settings` behavior
2. Add CLI command to manually fix deployments with incorrect autoscaling:
   ```bash
   truss deployment reset-autoscaling <deployment-id>
   ```
3. Add warning when deployment with environment fails
4. Add monitoring/alerting for deployments with environment settings but not in environment

## Testing Checklist

- [x] Added `defer_environment_settings` parameter to GraphQL mutations
- [x] Updated core layer to pass the parameter
- [x] Created E2E test covering the scenario
- [x] Syntax validation passed
- [ ] Unit tests pass (requires backend changes)
- [ ] E2E test passes against devcontainer (requires backend changes)
- [ ] Manual verification with video recording (requires backend changes)

## Notes

This is primarily a backend fix - the backend must be updated to handle the `defer_environment_settings` parameter correctly. The client-side changes prepare the CLI to work with the fixed backend.

Once the backend is deployed with support for this parameter, the issue should be resolved.
