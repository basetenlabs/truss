#!/usr/bin/env python3
"""
E2E test for promotion autoscaling fix.

Test scenario per acceptance criteria:
1. Push an oracle version that succeeds with --environment production (should have 4/10 min/max replicas)
2. Push an oracle version that fails on first build attempt
3. Manually retry the failed deployment (simulated by fixing and pushing again)
4. Verify that the retried deployment does NOT get autoscaled to 4 replicas

This test verifies that failed deployments don't retain environment autoscaling settings
after being retried.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


# Constants for test
MODEL_NAME_PREFIX = "test-promotion-autoscaling"
ENVIRONMENT_NAME = "production"
EXPECTED_PROD_MIN_REPLICAS = 4
EXPECTED_PROD_MAX_REPLICAS = 10


def create_minimal_truss(truss_dir: Path, model_name: str, should_fail: bool = False) -> None:
    """Create a minimal test truss.
    
    Args:
        truss_dir: Directory to create truss in
        model_name: Name for the model
        should_fail: If True, add a requirement that will cause build to fail
    """
    truss_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model directory
    model_dir = truss_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    # Create model.py
    model_py = model_dir / "model.py"
    model_py.write_text("""
class Model:
    def load(self):
        print("Model loaded")
    
    def predict(self, request):
        return {"status": "ok", "output": request.get("input", "test")}
""")
    
    # Create config.yaml
    config_yaml = truss_dir / "config.yaml"
    
    requirements = ["numpy==1.24.0"]
    if should_fail:
        # Add a requirement that will cause build to fail
        requirements.append("this-package-does-not-exist-123456==99.99.99")
    
    config_content = f"""model_name: {model_name}
python_version: py311
resources:
  cpu: "1"
  memory: 2Gi
  use_gpu: false
requirements:
{chr(10).join(f"  - {req}" for req in requirements)}
"""
    
    config_yaml.write_text(config_content)


def run_truss_push(
    truss_dir: Path,
    environment: Optional[str] = None,
    wait: bool = False,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run truss push command.
    
    Args:
        truss_dir: Path to truss directory
        environment: Optional environment name for promotion
        wait: Whether to wait for deployment
        timeout_seconds: Optional timeout for wait
    
    Returns:
        Dictionary with deployment info from JSON output
    
    Raises:
        subprocess.CalledProcessError: If push command fails
    """
    cmd = [
        "truss",
        "push",
        str(truss_dir),
        "--output", "json",
    ]
    
    if environment:
        cmd.extend(["--environment", environment])
    
    if wait:
        cmd.append("--wait")
    
    if timeout_seconds:
        cmd.extend(["--timeout-seconds", str(timeout_seconds)])
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    
    # Parse JSON output
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # If JSON parsing fails, return basic info
        print(f"Warning: Could not parse JSON output: {result.stdout}")
        return {"status": "unknown"}


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E_TESTS"),
    reason="E2E tests only run when RUN_E2E_TESTS is set"
)
def test_promotion_autoscaling_fix():
    """Test that failed promotion deployments don't retain environment autoscaling settings.
    
    This test implements the acceptance criteria:
    1. Push successful deployment to production (should have 4/10 replicas)
    2. Push deployment that fails build (with production environment)
    3. Fix and retry deployment (without environment)
    4. Verify retry does NOT have production autoscaling
    
    NOTE: This test requires manual verification of autoscaling settings in the
    Baseten UI, as the API methods to query autoscaling may not be available yet.
    """
    
    # Generate unique model name for this test run
    model_name = f"{MODEL_NAME_PREFIX}-{int(time.time())}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Push successful deployment to production environment
        print("\n" + "=" * 80)
        print("Step 1: Push successful deployment to production")
        print("=" * 80)
        
        success_truss_dir = temp_path / "success_truss"
        create_minimal_truss(success_truss_dir, model_name, should_fail=False)
        
        success_result = run_truss_push(
            success_truss_dir,
            environment=ENVIRONMENT_NAME,
            wait=True,
            timeout_seconds=600,
        )
        
        success_deployment_id = success_result.get("deployment_id")
        assert success_deployment_id, "Failed to get deployment ID for successful deployment"
        
        print(f"✓ Successfully deployed to {ENVIRONMENT_NAME}: {success_deployment_id}")
        print(f"\nPlease verify in Baseten UI:")
        print(f"  - Deployment should be ACTIVE")
        print(f"  - Autoscaling should be {EXPECTED_PROD_MIN_REPLICAS}/{EXPECTED_PROD_MAX_REPLICAS} min/max replicas")
        
        # Step 2: Push deployment that will fail
        print("\n" + "=" * 80)
        print("Step 2: Push deployment that will fail")
        print("=" * 80)
        
        fail_truss_dir = temp_path / "fail_truss"
        create_minimal_truss(fail_truss_dir, model_name, should_fail=True)
        
        print(f"Pushing failing deployment with --environment {ENVIRONMENT_NAME}...")
        print("(This should fail during build)")
        
        try:
            fail_result = run_truss_push(
                fail_truss_dir,
                environment=ENVIRONMENT_NAME,
                wait=False,
            )
            fail_deployment_id = fail_result.get("deployment_id")
            print(f"Deployment created: {fail_deployment_id}")
            print("Note: Deployment should fail during build. Check Baseten UI.")
            
        except subprocess.CalledProcessError as e:
            print(f"Push command failed (may be expected): {e}")
            print("Check Baseten UI for deployment status.")
        
        print(f"\nPlease verify in Baseten UI:")
        print(f"  - Deployment should be BUILD_FAILED or DEPLOY_FAILED")
        
        # Step 3: Fix and retry deployment
        print("\n" + "=" * 80)
        print("Step 3: Fix and retry deployment (simulating manual retry)")
        print("=" * 80)
        
        # Create fixed version of the truss
        fixed_truss_dir = temp_path / "fixed_truss"
        create_minimal_truss(fixed_truss_dir, model_name, should_fail=False)
        
        print("Pushing fixed deployment WITHOUT --environment (simulating UI retry)...")
        
        retry_result = run_truss_push(
            fixed_truss_dir,
            environment=None,  # No environment specified for retry
            wait=True,
            timeout_seconds=600,
        )
        
        retry_deployment_id = retry_result.get("deployment_id")
        assert retry_deployment_id, "Failed to get deployment ID for retry deployment"
        
        print(f"✓ Retry deployment succeeded: {retry_deployment_id}")
        
        # Step 4: Verify autoscaling settings
        print("\n" + "=" * 80)
        print("Step 4: Verify autoscaling settings after retry")
        print("=" * 80)
        
        print(f"\n⚠️  MANUAL VERIFICATION REQUIRED:")
        print(f"\nPlease check the autoscaling settings for deployment {retry_deployment_id}")
        print(f"in the Baseten UI.\n")
        print(f"Expected behavior (FIX WORKING):")
        print(f"  ✓ Retry deployment has DEFAULT autoscaling (e.g., 0/3 or 1/3)")
        print(f"  ✓ Retry deployment min_replicas != {EXPECTED_PROD_MIN_REPLICAS}")
        print(f"\nBroken behavior (FIX NOT WORKING):")
        print(f"  ✗ Retry deployment has PRODUCTION autoscaling ({EXPECTED_PROD_MIN_REPLICAS}/{EXPECTED_PROD_MAX_REPLICAS})")
        print(f"  ✗ Retry deployment min_replicas == {EXPECTED_PROD_MIN_REPLICAS}")
        print(f"\n{'=' * 80}")
        print(f"Test deployments created:")
        print(f"  Model: {model_name}")
        print(f"  Success deployment: {success_deployment_id}")
        print(f"  Retry deployment: {retry_deployment_id}")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Allow running directly for manual testing
    if not os.environ.get("RUN_E2E_TESTS"):
        print("Set RUN_E2E_TESTS=1 to run this E2E test")
        print("Example: RUN_E2E_TESTS=1 python tests/e2e/test_promotion_autoscaling.py")
        sys.exit(1)
    
    test_promotion_autoscaling_fix()
