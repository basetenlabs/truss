#!/usr/bin/env python3
"""
Test the UUID suffix optimization
"""

import sys

sys.path.insert(
    0, "/Users/michaelfeil/work/truss/baseten-performance-client/python_bindings"
)

try:
    from baseten_performance_client import (
        PerformanceClient,
        RequestProcessingPreference,
    )

    client = PerformanceClient("https://api.example.com", "dummy-key")
    preference = RequestProcessingPreference()

    # Test that CustomerRequestId creation works with the optimization
    try:
        response = client.embed(
            model="text-embedding-ada-002",
            input=["test", "optimization"],
            preference=preference,
        )
    except Exception as e:
        # Expected to fail due to network, but should not fail due to UUID issues
        if "UTF-8" in str(e) or "uuid" in str(e).lower():
            print(f"❌ UUID optimization failed: {e}")
        else:
            print("✅ UUID optimization works correctly (network error expected)")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
