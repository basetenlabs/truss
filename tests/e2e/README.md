# E2E Tests

End-to-end tests that require actual deployments to Baseten.

## Running E2E Tests

E2E tests are skipped by default. To run them, set the `RUN_E2E_TESTS` environment variable:

```bash
export RUN_E2E_TESTS=1
pytest tests/e2e/ -xvs
```

Or run a specific test:

```bash
RUN_E2E_TESTS=1 pytest tests/e2e/test_promotion_autoscaling.py -xvs
```

## Prerequisites

- Configured Baseten remote with valid API key
- Access to create deployments
- Access to create/use environments
- Sufficient deployment quota

## Available Tests

### test_promotion_autoscaling.py

Tests the fix for the promotion autoscaling bug where failed deployments
retain environment autoscaling settings after retry.

**Test scenario**:
1. Push successful deployment to production (should have 4/10 min/max replicas)
2. Push deployment that fails build (with production environment)
3. Fix and retry deployment (without environment)
4. Verify retry does NOT have production autoscaling

**Duration**: ~10-15 minutes (includes build + deploy times)

## Manual Testing

For manual testing with video recording, use the provided script:

```bash
./scripts/manual_e2e_test.sh
```

This script guides you through the E2E test scenario step-by-step with
prompts to verify results in the Baseten UI.

## Tips

- Use unique model names for each test run to avoid conflicts
- Clean up test deployments after testing to avoid quota issues
- For debugging, use `--output json` with truss push to get structured output
- Check deployment status in Baseten UI if tests fail unexpectedly

## Troubleshooting

**Test times out waiting for deployment**:
- Increase timeout_seconds in test
- Check deployment status in Baseten UI
- Verify API key and permissions

**Deployment fails unexpectedly**:
- Check build logs in Baseten UI
- Verify requirements are valid
- Ensure sufficient resources/quota

**Can't get deployment info**:
- Verify API methods are available
- Check API key permissions
- Ensure model and deployment exist
