#!/usr/bin/env python3
"""Verify connectivity and functionality against a live dp_worker.

Usage (from a pod on the same cluster):
    pip install httpx
    python verify_worker.py <worker_url>

Example:
    python verify_worker.py http://baseten-training-job-q40rr9w-multinode-0.baseten-training-job-q40rr9w-multinode.org-9914a591b6e04ff7848a21ae64fa3398.svc.cluster.local:8001
"""

import json
import sys
import time

import httpx

TIMEOUT = 300.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_worker.py <worker_url>")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    client = httpx.Client(timeout=TIMEOUT)

    # 1. Health check (with retry — worker may still be booting)
    print(f"[1/6] Health check: {base_url}/health")
    for attempt in range(60):
        try:
            resp = client.get(f"{base_url}/health")
            if resp.status_code == 200:
                print("  OK")
                break
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        print(f"  waiting for worker (attempt {attempt + 1}/60)...")
        time.sleep(5)
    else:
        print("  FAILED: worker did not become healthy after 5 minutes")
        sys.exit(1)

    # 2. Forward-backward
    print("[2/6] forward_backward (2 samples, cross_entropy)")
    resp = client.post(f"{base_url}/forward_backward", json={
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(10))}]},
                "loss_fn_inputs": {"reward": {"data": [1.0], "dtype": "float32", "shape": [1]}},
            },
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(10, 20))}]},
                "loss_fn_inputs": {"reward": {"data": [0.5], "dtype": "float32", "shape": [1]}},
            },
        ],
        "loss_fn": "cross_entropy",
    })
    assert resp.status_code == 200, f"forward_backward failed: {resp.status_code} {resp.text}"
    result = resp.json()
    print(f"  loss={result.get('metrics', {}).get('loss', '?')}")

    # 3. Second forward-backward (gradient accumulation)
    print("[3/6] forward_backward (gradient accumulation step 2)")
    resp = client.post(f"{base_url}/forward_backward", json={
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(20, 30))}]},
                "loss_fn_inputs": {"reward": {"data": [1.0], "dtype": "float32", "shape": [1]}},
            },
        ],
        "loss_fn": "cross_entropy",
    })
    assert resp.status_code == 200, f"forward_backward failed: {resp.status_code} {resp.text}"
    result = resp.json()
    print(f"  loss={result.get('metrics', {}).get('loss', '?')}")

    # 4. Optimizer step
    print("[4/6] optim_step")
    resp = client.post(f"{base_url}/optim_step", json={})
    assert resp.status_code == 200, f"optim_step failed: {resp.status_code} {resp.text}"
    result = resp.json()
    print(f"  step={result.get('metrics', {}).get('step', '?')}, lr={result.get('metrics', {}).get('lr', '?')}")

    # 5. Switch to inference
    print("[5/6] to_inference (sync weights to vLLM)")
    resp = client.post(f"{base_url}/to_inference")
    assert resp.status_code == 200, f"to_inference failed: {resp.status_code} {resp.text}"
    result = resp.json()
    print(f"  mode={result.get('mode', '?')}")

    # 6. Sample
    print("[6/6] sample (generate text)")
    resp = client.post(f"{base_url}/sample", json={
        "inputs": [
            {
                "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
                "max_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        ],
    })
    assert resp.status_code == 200, f"sample failed: {resp.status_code} {resp.text}"
    result = resp.json()
    outputs = result.get("outputs", [])
    if outputs:
        print(f"  generated: {outputs[0].get('generated_text', '')[:200]}")
    else:
        print(f"  raw: {json.dumps(result)[:200]}")

    print("\nAll checks passed!")
    client.close()


if __name__ == "__main__":
    main()
