#!/usr/bin/env python3
"""End-to-end test: deploy a training job and verify the full loop.

Usage:
    python e2e.py --namespace org-xxx [--remote baseten] [--model Qwen/Qwen3-8B]

What it does:
    1. Deploys a training job via create_training_client()
    2. Deploys a verification pod into the given namespace
    3. The pod waits for the worker to boot, then runs:
       health → forward_backward x2 → optim_step → save_weights → sample
    4. Streams pod logs and reports pass/fail
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add trainers to path for local dev
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainers"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "truss-train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


VERIFY_SCRIPT = r'''
import json
import sys
import time
import httpx

TIMEOUT = 300.0
base_url = sys.argv[1]
client = httpx.Client(timeout=TIMEOUT)

# 1. Health (with retry)
print("[1/6] Health check")
for attempt in range(60):
    try:
        if client.get(f"{base_url}/health").status_code == 200:
            print("  OK")
            break
    except (httpx.ConnectError, httpx.ConnectTimeout):
        pass
    print(f"  waiting ({attempt + 1}/60)...")
    time.sleep(5)
else:
    print("  FAILED: worker not healthy after 5 min")
    sys.exit(1)

# 2. forward_backward
print("[2/6] forward_backward")
resp = client.post(f"{base_url}/forward_backward", json={
    "data": [
        {"model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(10))}]},
         "loss_fn_inputs": {"reward": {"data": [1.0], "dtype": "float32", "shape": [1]}}},
        {"model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(10, 20))}]},
         "loss_fn_inputs": {"reward": {"data": [0.5], "dtype": "float32", "shape": [1]}}},
    ],
    "loss_fn": "cross_entropy",
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
print(f"  loss={resp.json().get('metrics', {}).get('loss', '?')}")

# 3. forward_backward (grad accum)
print("[3/6] forward_backward (grad accum)")
resp = client.post(f"{base_url}/forward_backward", json={
    "data": [
        {"model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(20, 30))}]},
         "loss_fn_inputs": {"reward": {"data": [1.0], "dtype": "float32", "shape": [1]}}},
    ],
    "loss_fn": "cross_entropy",
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
print(f"  loss={resp.json().get('metrics', {}).get('loss', '?')}")

# 4. optim_step with adam_params
print("[4/6] optim_step")
resp = client.post(f"{base_url}/optim_step", json={
    "adam_params": {"learning_rate": 5e-6, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0, "grad_clip_norm": 0.0}
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
r = resp.json()
print(f"  step={r.get('metrics', {}).get('step', '?')}, lr={r.get('metrics', {}).get('lr', '?')}")

# 5. to_inference
print("[5/6] save_weights_and_get_sampling_client")
resp = client.post(f"{base_url}/to_inference")
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
print(f"  mode={resp.json().get('mode', '?')}")

# 6. sample
print("[6/6] sample")
resp = client.post(f"{base_url}/sample", json={
    "inputs": [{"messages": [{"role": "user", "content": "What is 2+2? One word."}], "max_tokens": 64, "temperature": 0.0, "top_p": 1.0}],
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
outputs = resp.json().get("outputs", [])
if outputs:
    print(f"  generated: {outputs[0].get('generated_text', '')[:200]}")

print("\nAll checks passed!")
'''


def kubectl(*args, capture=False):
    cmd = ["kubectl"] + list(args)
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd)


def deploy_job(namespace, model, accelerator, gpu_count, remote, workspace_root):
    from trainers import create_training_client

    client = create_training_client(
        base_model=model,
        worker_url="",
        gpu_count=gpu_count,
        accelerator=accelerator,
        training_gpus=[0],
        inference_gpus=[1] if gpu_count > 1 else [0],
        max_seq_len=4096,
        worker_port=8001,
        namespace=namespace,
        remote=remote,
        workspace_root=Path(workspace_root) if workspace_root else None,
    )
    # Extract job_id from the URL
    # URL: http://baseten-training-job-{job_id}-multinode-0...
    url = client._base_url
    job_id = url.split("baseten-training-job-")[1].split("-multinode")[0]
    return job_id, url


def run_verify_pod(namespace, worker_url, job_id):
    pod_name = f"e2e-verify-{job_id}"
    configmap_name = f"e2e-script-{job_id}"

    # Write script to a temp file and create configmap
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(VERIFY_SCRIPT)
        script_path = f.name

    print(f"\nCreating configmap {configmap_name}...")
    kubectl("create", "configmap", configmap_name,
            "-n", namespace,
            f"--from-file=verify.py={script_path}")

    pod_yaml = json.dumps({
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": namespace},
        "spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "test",
                "image": "python:3.12-slim",
                "command": ["sh", "-c", f"pip install httpx && python /test/verify.py {worker_url}"],
                "volumeMounts": [{"name": "script", "mountPath": "/test"}],
            }],
            "volumes": [{"name": "script", "configMap": {"name": configmap_name}}],
        },
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(pod_yaml)
        pod_path = f.name

    print(f"Creating pod {pod_name}...")
    kubectl("apply", "-f", pod_path)

    print(f"Streaming logs (waiting for pod to start)...\n")
    time.sleep(3)

    result = kubectl("logs", "-n", namespace, pod_name, "-f", capture=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Check pod exit code
    status = kubectl("get", "pod", pod_name, "-n", namespace,
                     "-o", "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}",
                     capture=True)
    exit_code = status.stdout.strip()

    # Cleanup
    print(f"\nCleaning up {pod_name}...")
    kubectl("delete", "pod", pod_name, "-n", namespace, "--ignore-not-found")
    kubectl("delete", "configmap", configmap_name, "-n", namespace, "--ignore-not-found")

    if exit_code == "0":
        print("\n=== E2E TEST PASSED ===")
        return 0
    else:
        print(f"\n=== E2E TEST FAILED (exit code: {exit_code}) ===")
        return 1


def main():
    parser = argparse.ArgumentParser(description="End-to-end trainers test")
    parser.add_argument("--namespace", required=True, help="K8s namespace (e.g. org-xxx)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model to train")
    parser.add_argument("--accelerator", default="H200", help="GPU type")
    parser.add_argument("--gpu-count", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--remote", default="baseten", help="Baseten remote name")
    parser.add_argument("--workspace-root", default=None, help="Override server workspace path")
    parser.add_argument("--job-id", default=None, help="Skip deploy, use existing job")
    args = parser.parse_args()

    if args.job_id:
        job_id = args.job_id
        worker_url = (
            f"http://baseten-training-job-{job_id}-multinode-0"
            f".baseten-training-job-{job_id}-multinode"
            f".{args.namespace}.svc.cluster.local:8001"
        )
        print(f"Using existing job: {job_id}")
    else:
        print(f"Deploying {args.model} on {args.gpu_count}x {args.accelerator}...")
        job_id, worker_url = deploy_job(
            args.namespace, args.model, args.accelerator,
            args.gpu_count, args.remote, args.workspace_root,
        )

    print(f"Job ID: {job_id}")
    print(f"Worker URL: {worker_url}")

    sys.exit(run_verify_pod(args.namespace, worker_url, job_id))


if __name__ == "__main__":
    main()
