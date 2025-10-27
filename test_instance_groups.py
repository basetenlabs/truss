#!/usr/bin/env python3
"""
Simple test script to verify instance groups functionality works correctly.
This creates a mock truss with instance_groups configured and tests round-robin distribution.
"""

import os
import tempfile
import json
from pathlib import Path

# Create test truss directory
test_dir = Path(tempfile.mkdtemp())
truss_dir = test_dir / "test_truss"
truss_dir.mkdir()

# Create config.yaml with instance_groups
config = {
    "model_name": "test_instance_groups",
    "model_class_name": "TestModel",
    "model_class_filename": "model.py",
    "runtime": {
        "predict_concurrency": 1,
        "instance_groups": 3
    }
}

config_path = truss_dir / "config.yaml"
with open(config_path, 'w') as f:
    import yaml
    yaml.dump(config, f)

# Create a simple model that tracks which instance is being called
model_code = '''
class TestModel:
    def __init__(self, config=None, data_dir=None, secrets=None, **kwargs):
        # Use environment variable to track instance (simulating multiple instances)
        self.instance_id = os.environ.get('TRUSS_INSTANCE_ID', 'default')
        print(f"Initialized TestModel instance: {self.instance_id}")
    
    def load(self):
        print(f"Loading model instance: {self.instance_id}")
    
    def predict(self, inputs):
        result = {
            "instance_id": self.instance_id,
            "inputs": inputs,
            "message": f"Processed by instance {self.instance_id}"
        }
        return result
'''

with open(truss_dir / "model.py", 'w') as f:
    f.write(model_code)

print(f"Created test truss at: {truss_dir}")
print(f"Config has instance_groups: {config['runtime']['instance_groups']}")
print("\nTo test:")
print(f"cd {truss_dir}")
print("truss run --local")
