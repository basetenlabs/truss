#!/bin/bash
# Manual E2E test script for promotion autoscaling fix
# Usage: ./scripts/manual_e2e_test.sh

set -e

# Configuration
MODEL_NAME="test-promotion-autoscaling-$(date +%s)"
ENVIRONMENT="production"
TEST_DIR="/tmp/truss_manual_e2e_test_$(date +%s)"

echo "========================================="
echo "Manual E2E Test: Promotion Autoscaling Fix"
echo "========================================="
echo ""
echo "Model Name: $MODEL_NAME"
echo "Environment: $ENVIRONMENT"
echo "Test Directory: $TEST_DIR"
echo ""

# Create test directory
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Step 1: Create and push successful deployment to production
echo "========================================="
echo "Step 1: Create successful production deployment"
echo "========================================="

mkdir -p success_truss/model
cat > success_truss/model/model.py << 'EOF'
class Model:
    def load(self):
        print("Model loaded successfully")
    
    def predict(self, request):
        return {"status": "ok", "output": "success"}
EOF

cat > success_truss/config.yaml << EOF
model_name: $MODEL_NAME
python_version: py311
resources:
  cpu: "1"
  memory: 2Gi
  use_gpu: false
requirements:
  - numpy==1.24.0
EOF

echo "Pushing successful deployment to $ENVIRONMENT..."
truss push success_truss --environment "$ENVIRONMENT" --wait

echo ""
echo "✓ Step 1 complete. Check Baseten UI:"
echo "  - Deployment should be ACTIVE"
echo "  - Autoscaling should be 4/10 min/max replicas"
echo ""
read -p "Press Enter to continue to Step 2..."

# Step 2: Create deployment that will fail
echo ""
echo "========================================="
echo "Step 2: Create deployment that will fail"
echo "========================================="

mkdir -p fail_truss/model
cat > fail_truss/model/model.py << 'EOF'
class Model:
    def load(self):
        print("Model loaded")
    
    def predict(self, request):
        return {"status": "ok"}
EOF

cat > fail_truss/config.yaml << EOF
model_name: $MODEL_NAME
python_version: py311
resources:
  cpu: "1"
  memory: 2Gi
  use_gpu: false
requirements:
  - numpy==1.24.0
  - this-package-does-not-exist-12345==99.99.99
EOF

echo "Pushing failing deployment with --environment $ENVIRONMENT..."
echo "(This should fail during build)"

set +e  # Don't exit on error
truss push fail_truss --environment "$ENVIRONMENT"
PUSH_EXIT_CODE=$?
set -e

if [ $PUSH_EXIT_CODE -eq 0 ]; then
    echo "Note: Push command succeeded, but deployment should fail during build"
else
    echo "Push command returned error code $PUSH_EXIT_CODE (may be expected)"
fi

echo ""
echo "✓ Step 2 complete. Check Baseten UI:"
echo "  - Deployment should be BUILD_FAILED or DEPLOY_FAILED"
echo "  - Note the deployment ID"
echo ""
read -p "Press Enter to continue to Step 3..."

# Step 3: Fix and retry without environment
echo ""
echo "========================================="
echo "Step 3: Fix and retry deployment"
echo "========================================="

mkdir -p fixed_truss/model
cat > fixed_truss/model/model.py << 'EOF'
class Model:
    def load(self):
        print("Model loaded successfully")
    
    def predict(self, request):
        return {"status": "ok", "output": "fixed"}
EOF

cat > fixed_truss/config.yaml << EOF
model_name: $MODEL_NAME
python_version: py311
resources:
  cpu: "1"
  memory: 2Gi
  use_gpu: false
requirements:
  - numpy==1.24.0
EOF

echo "Pushing fixed deployment WITHOUT --environment (simulating UI retry)..."
truss push fixed_truss --wait

echo ""
echo "✓ Step 3 complete. Check Baseten UI:"
echo "  - Deployment should be ACTIVE"
echo "  - Check autoscaling settings"
echo ""
echo "========================================="
echo "Step 4: Verify Results"
echo "========================================="
echo ""
echo "Expected behavior (FIX WORKING):"
echo "  ✓ Retry deployment has DEFAULT autoscaling (e.g., 0/3 or 1/3)"
echo "  ✓ Retry deployment min_replicas != 4"
echo ""
echo "Broken behavior (FIX NOT WORKING):"
echo "  ✗ Retry deployment has PRODUCTION autoscaling (4/10)"
echo "  ✗ Retry deployment min_replicas == 4"
echo ""
read -p "What are the autoscaling settings? (min/max): " MIN_REPLICAS MAX_REPLICAS

if [ "$MIN_REPLICAS" = "4" ]; then
    echo ""
    echo "✗✗✗ TEST FAILED ✗✗✗"
    echo "Retry deployment has production autoscaling (min_replicas=4)"
    echo "The fix is NOT working correctly."
    exit 1
else
    echo ""
    echo "✓✓✓ TEST PASSED ✓✓✓"
    echo "Retry deployment does NOT have production autoscaling (min_replicas=$MIN_REPLICAS)"
    echo "The fix is working correctly!"
fi

# Cleanup
echo ""
read -p "Clean up test directory $TEST_DIR? (y/n): " CLEANUP
if [ "$CLEANUP" = "y" ]; then
    rm -rf "$TEST_DIR"
    echo "✓ Cleaned up test directory"
fi

echo ""
echo "Manual E2E test complete!"
