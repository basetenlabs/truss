#!/usr/bin/env python
import json
import sys

RESULTS = []


def log(message, status="INFO"):
    prefix = {"INFO": "  ", "PASS": "✅", "FAIL": "❌"}[status]
    print(f"{prefix} {message}")


def run_test(name, test_fn):
    try:
        result = test_fn()
        if result:
            log(f"{name}: {result}", "PASS")
            RESULTS.append({"test": name, "status": "pass", "detail": result})
        else:
            log(f"{name}: returned falsy", "FAIL")
            RESULTS.append({"test": name, "status": "fail", "detail": "returned falsy"})
    except Exception as e:
        log(f"{name}: {e}", "FAIL")
        RESULTS.append({"test": name, "status": "fail", "detail": str(e)})


def run_tests_and_exit(title, tests):
    print("=" * 60)
    print(title)
    print("=" * 60)

    for name, fn in tests:
        run_test(name, fn)

    passed = sum(1 for r in RESULTS if r["status"] == "pass")
    failed = len(RESULTS) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    print("\nJSON Results:")
    print(json.dumps(RESULTS, indent=2))
    sys.exit(0 if failed == 0 else 1)
