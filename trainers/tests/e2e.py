#!/usr/bin/env python3
"""End-to-end test: deploy a training job and run tests on-cluster.

Usage:
    python e2e.py --namespace org-xxx --test smoke
    python e2e.py --namespace org-xxx --test math_rl
    python e2e.py --namespace org-xxx --test multiturn_rl
    python e2e.py --namespace org-xxx --test math_rl --job-id q86dgg3  # skip deploy

Tests:
    smoke        — health, forward_backward, optim_step, save_weights, sample
    math_rl      — GRPO on arithmetic problems (single-turn RL)
    multiturn_rl — number guessing game (multi-turn RL)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainers"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "truss-train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Test scripts (embedded, run inside cluster pods) ──────────────

SMOKE_SCRIPT = r'''
import sys, time, httpx
TIMEOUT = 300.0
base_url = sys.argv[1]
client = httpx.Client(timeout=TIMEOUT)
adam = {"learning_rate": 5e-6, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12,
        "weight_decay": 0.0, "grad_clip_norm": 0.0}

# ── [1/7] Health ────────────────────────────────────────────────────
print("[1/7] Health check")
for attempt in range(60):
    try:
        if client.get(f"{base_url}/health").status_code == 200:
            print("  OK"); break
    except (httpx.ConnectError, httpx.ConnectTimeout): pass
    print(f"  waiting ({attempt + 1}/60)..."); time.sleep(5)
else:
    print("  FAILED: worker not healthy after 5 min"); sys.exit(1)

# ── [2/7] cross_entropy sanity ──────────────────────────────────────
print("[2/7] forward_backward (cross_entropy)")
resp = client.post(f"{base_url}/forward_backward", json={
    "data": [
        {"model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(1, 12))}]},
         "loss_fn_inputs": {}},
        {"model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(5, 16))}]},
         "loss_fn_inputs": {}},
    ],
    "loss_fn": "cross_entropy",
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
r = resp.json()
assert "loss" in r.get("metrics", {}), f"missing loss in metrics: {r}"
assert r.get("loss_fn_outputs") and "loss" in r["loss_fn_outputs"][0], \
    f"expected loss_fn_outputs[0]['loss'], got: {r.get('loss_fn_outputs')}"
print(f"  loss={r['metrics']['loss']:.4f}")

# ── [3/7] optim_step ────────────────────────────────────────────────
print("[3/7] optim_step")
resp = client.post(f"{base_url}/optim_step", json={"adam_params": adam})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
assert resp.json()["metrics"]["step"] == 1, f"expected step=1, got: {resp.json()}"
print(f"  step={resp.json()['metrics']['step']}")

# ── [4/7] to_inference ──────────────────────────────────────────────
print("[4/7] to_inference")
resp = client.post(f"{base_url}/to_inference")
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
assert resp.json()["mode"] == "inference", f"expected mode=inference: {resp.json()}"
print("  mode=inference")

# ── [5/7] sample — verify logprobs returned ──────────────────────────
print("[5/7] sample (verify logprobs)")
prompt_tokens = list(range(1, 9))  # 8 tokens as prompt
resp = client.post(f"{base_url}/sample", json={
    "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
    "num_samples": 1,
    "sampling_params": {"max_tokens": 16, "temperature": 1.0, "top_p": 1.0},
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
seqs = resp.json()["sequences"]
assert seqs, "no sequences returned"
seq = seqs[0]
gen_tokens = seq["tokens"]
gen_logprobs = seq["logprobs"]
assert gen_logprobs is not None, "logprobs not returned from sample — vLLM logprobs not wired up"
assert len(gen_logprobs) == len(gen_tokens), \
    f"logprob/token length mismatch: {len(gen_logprobs)} vs {len(gen_tokens)}"
assert all(lp <= 0.0 for lp in gen_logprobs), \
    f"logprobs must be <= 0 (log probabilities), got: {gen_logprobs}"
print(f"  gen_tokens={gen_tokens[:8]}")
print(f"  gen_logprobs={[f'{lp:.3f}' for lp in gen_logprobs[:8]]}")

# ── [6/7] importance_sampling forward_backward ───────────────────────
# Construct the datum exactly as the training loop does:
#   model_input  = prompt_tokens + gen_tokens[:-1]
#   target_tokens = [0]*ob_len + gen_tokens          (0 = no loss at prompt positions)
#   logprobs      = [0.0]*ob_len + gen_logprobs      (old-policy logprobs)
#   advantages    = [0.0]*ob_len + [adv]*comp_len
print("[6/7] forward_backward (importance_sampling)")
ob_len = len(prompt_tokens) - 1   # prompt positions with no loss
model_input_tokens = prompt_tokens + gen_tokens[:-1]
target_tokens = [0] * ob_len + gen_tokens
padded_logprobs = [0.0] * ob_len + gen_logprobs
padded_advantages = [0.0] * ob_len + [0.5] * len(gen_tokens)
T = len(target_tokens)

resp = client.post(f"{base_url}/forward_backward", json={
    "data": [{
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": model_input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "dtype": "int64",   "shape": [T]},
            "logprobs":      {"data": padded_logprobs, "dtype": "float32", "shape": [T]},
            "advantages":    {"data": padded_advantages, "dtype": "float32", "shape": [T]},
        },
    }],
    "loss_fn": "importance_sampling",
})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
r = resp.json()
assert r.get("loss_fn_output_type") == "importance_sampling", \
    f"expected loss_fn_output_type=importance_sampling: {r}"
assert "loss" in r.get("metrics", {}), f"missing loss in metrics: {r}"
assert r.get("loss_fn_outputs") and "loss" in r["loss_fn_outputs"][0], \
    f"expected loss_fn_outputs[0]['loss'], got: {r.get('loss_fn_outputs')}"
print(f"  loss={r['metrics']['loss']:.4f}")
print(f"  per_sample_loss={r['loss_fn_outputs'][0]['loss']['data']}")

# ── [7/7] optim_step ────────────────────────────────────────────────
print("[7/7] optim_step")
resp = client.post(f"{base_url}/optim_step", json={"adam_params": adam})
assert resp.status_code == 200, f"FAILED: {resp.status_code} {resp.text}"
assert resp.json()["metrics"]["step"] == 2, f"expected step=2, got: {resp.json()}"
print(f"  step={resp.json()['metrics']['step']}")

print("\nAll checks passed!")
'''

MATH_RL_SCRIPT = r'''
import random, re, sys, time, json, httpx
from transformers import AutoTokenizer

base_url = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
STEPS, GROUP_SIZE, PROBLEMS, LR, MAX_TOKENS = 3, 4, 2, 1e-5, 16
TIMEOUT = 300.0

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
client = httpx.Client(timeout=TIMEOUT)
rng = random.Random(42)

# Wait for worker
print("[0] Waiting for worker...")
for attempt in range(60):
    try:
        if client.get(f"{base_url}/health").status_code == 200: break
    except (httpx.ConnectError, httpx.ConnectTimeout): pass
    time.sleep(5)
else:
    print("FAILED: worker not healthy"); sys.exit(1)
print("  Worker ready\n")

adam = {"learning_rate": LR, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.0}
sparams = {"max_tokens": MAX_TOKENS, "temperature": 1.0, "top_p": 0.95}

for step in range(STEPS):
    t0 = time.perf_counter()
    print(f"--- Step {step+1}/{STEPS} ---")

    # Switch to inference
    resp = client.post(f"{base_url}/to_inference")
    assert resp.status_code == 200, f"to_inference failed: {resp.text}"

    all_data = []
    correct = 0
    total = 0

    for p in range(PROBLEMS):
        x, y = rng.randint(0, 100), rng.randint(0, 100)
        prompt = f"What is {x} + {y}? Answer with just the number.\n"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        rewards = []
        completions = []

        for s in range(GROUP_SIZE):
            resp = client.post(f"{base_url}/sample", json={
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
                "num_samples": 1, "sampling_params": sparams,
            })
            assert resp.status_code == 200, f"sample failed: {resp.text}"
            gen_tokens = resp.json()["sequences"][0]["tokens"]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            # Check answer
            reward = 0.0
            for word in gen_text.split():
                try:
                    if int(word) == x + y: reward = 1.0; break
                except ValueError: pass
            rewards.append(reward)
            completions.append(gen_tokens)
            correct += int(reward > 0)
            total += 1

        # GRPO advantages
        mean_r = sum(rewards) / len(rewards)
        std_r = max((sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5, 1e-8)
        advantages = [(r - mean_r) / std_r for r in rewards]

        for gen_tokens, adv in zip(completions, advantages):
            full = prompt_tokens + gen_tokens
            if len(full) < 2: continue
            all_data.append({
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": full}]},
                "loss_fn_inputs": {"advantages": {"data": [adv], "dtype": "float32", "shape": [1]}},
            })

    print(f"  Sampled {total}, accuracy={correct/total:.0%}")

    # Train
    if all_data:
        resp = client.post(f"{base_url}/forward_backward", json={"data": all_data, "loss_fn": "cross_entropy"})
        assert resp.status_code == 200, f"fwd_bwd failed: {resp.text}"
        print(f"  forward_backward: loss={resp.json().get('metrics',{}).get('loss','?')}")

        resp = client.post(f"{base_url}/optim_step", json={"adam_params": adam})
        assert resp.status_code == 200, f"optim failed: {resp.text}"
        print(f"  optim_step: step={resp.json().get('metrics',{}).get('step','?')}")

    print(f"  Done in {time.perf_counter()-t0:.1f}s\n")

print("=== Math RL acceptance test PASSED ===")
'''

MULTITURN_RL_SCRIPT = r'''
import random, re, sys, time, json, httpx
from transformers import AutoTokenizer

base_url = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
STEPS, GROUP_SIZE, PROBLEMS, MAX_TURNS, LR, MAX_TOKENS = 2, 4, 2, 5, 1e-5, 32
TIMEOUT = 300.0

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
client = httpx.Client(timeout=TIMEOUT)
rng = random.Random(42)

print("[0] Waiting for worker...")
for attempt in range(60):
    try:
        if client.get(f"{base_url}/health").status_code == 200: break
    except (httpx.ConnectError, httpx.ConnectTimeout): pass
    time.sleep(5)
else:
    print("FAILED: worker not healthy"); sys.exit(1)
print("  Worker ready\n")

adam = {"learning_rate": LR, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.0}
sparams = {"max_tokens": MAX_TOKENS, "temperature": 1.0, "top_p": 0.95}

for step in range(STEPS):
    t0 = time.perf_counter()
    print(f"--- Step {step+1}/{STEPS} ---")
    resp = client.post(f"{base_url}/to_inference")
    assert resp.status_code == 200, f"to_inference failed: {resp.text}"

    all_data = []
    solved = 0
    total = 0
    total_turns = 0

    for p in range(PROBLEMS):
        secret = rng.randint(1, 100)
        rewards = []
        trajectories = []

        for g in range(GROUP_SIZE):
            history = f"I'm thinking of a number between 1 and 100. You have {MAX_TURNS} guesses. Guess a number.\n"
            all_tokens = tokenizer.encode(history, add_special_tokens=False)
            game_solved = False
            turns = 0

            for turn in range(MAX_TURNS):
                turns += 1
                resp = client.post(f"{base_url}/sample", json={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": all_tokens}]},
                    "num_samples": 1, "sampling_params": sparams,
                })
                assert resp.status_code == 200, f"sample failed: {resp.text}"
                gen_tokens = resp.json()["sequences"][0]["tokens"]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                all_tokens = all_tokens + gen_tokens

                # Parse guess
                guess = None
                for word in re.findall(r'\d+', gen_text):
                    try: guess = int(word); break
                    except ValueError: pass

                if guess == secret:
                    feedback = f"Correct! The number was {secret}.\n"
                    game_solved = True
                elif guess is None:
                    feedback = f"I couldn't understand. Please guess a number.\n"
                elif guess < secret:
                    feedback = f"{guess} is too low. Guess higher.\n"
                else:
                    feedback = f"{guess} is too high. Guess lower.\n"

                all_tokens = all_tokens + tokenizer.encode(feedback, add_special_tokens=False)
                if game_solved: break

            rewards.append(1.0 if game_solved else 0.0)
            trajectories.append(all_tokens)
            solved += int(game_solved)
            total += 1
            total_turns += turns

        # GRPO advantages
        mean_r = sum(rewards) / len(rewards)
        std_r = max((sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5, 1e-8)
        advantages = [(r - mean_r) / std_r for r in rewards]

        for traj, adv in zip(trajectories, advantages):
            if len(traj) < 2: continue
            all_data.append({
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": traj}]},
                "loss_fn_inputs": {"advantages": {"data": [adv], "dtype": "float32", "shape": [1]}},
            })

    print(f"  {total} rollouts: solve_rate={solved/total:.0%}, avg_turns={total_turns/total:.1f}")

    if all_data:
        resp = client.post(f"{base_url}/forward_backward", json={"data": all_data, "loss_fn": "cross_entropy"})
        assert resp.status_code == 200, f"fwd_bwd failed: {resp.text}"
        print(f"  forward_backward: loss={resp.json().get('metrics',{}).get('loss','?')}")
        resp = client.post(f"{base_url}/optim_step", json={"adam_params": adam})
        assert resp.status_code == 200, f"optim failed: {resp.text}"
        print(f"  optim_step: step={resp.json().get('metrics',{}).get('step','?')}")

    print(f"  Done in {time.perf_counter()-t0:.1f}s\n")

print("=== Multi-turn RL acceptance test PASSED ===")
'''

TESTS = {
    "smoke": {"script": SMOKE_SCRIPT, "image": "python:3.12-slim", "pip": "httpx"},
    "math_rl": {"script": MATH_RL_SCRIPT, "image": "python:3.12-slim", "pip": "httpx transformers"},
    "multiturn_rl": {"script": MULTITURN_RL_SCRIPT, "image": "python:3.12-slim", "pip": "httpx transformers"},
}

# ── Infrastructure ────────────────────────────────────────────────

def kubectl(*args, capture=False):
    cmd = ["kubectl"] + list(args)
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd)


def deploy_job(namespace, model, accelerator, gpu_count, remote, workspace_root):
    from trainers import create_training_client
    client = create_training_client(
        base_model=model, worker_url="",
        gpu_count=gpu_count, accelerator=accelerator,
        training_gpus=[0], inference_gpus=[1] if gpu_count > 1 else [0],
        max_seq_len=4096, worker_port=8001, namespace=namespace,
        remote=remote,
        workspace_root=Path(workspace_root) if workspace_root else None,
    )
    url = client._base_url
    job_id = url.split("baseten-training-job-")[1].split("-multinode")[0]
    return job_id, url


def run_test_pod(namespace, worker_url, job_id, test_name, model):
    test_config = TESTS[test_name]
    safe_name = test_name.replace("_", "-")
    pod_name = f"e2e-{safe_name}-{job_id}"
    cm_name = f"e2e-script-{safe_name}-{job_id}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_config["script"])
        script_path = f.name

    print(f"\nCreating configmap {cm_name}...")
    kubectl("create", "configmap", cm_name, "-n", namespace,
            f"--from-file=test.py={script_path}")

    cmd = f"pip install {test_config['pip']} && python /test/test.py {worker_url} {model}"
    pod_spec = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": namespace},
        "spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "test", "image": test_config["image"],
                "command": ["sh", "-c", cmd],
                "volumeMounts": [{"name": "script", "mountPath": "/test"}],
                "resources": {"requests": {"memory": "4Gi", "cpu": "2"}, "limits": {"memory": "8Gi"}},
            }],
            "volumes": [{"name": "script", "configMap": {"name": cm_name}}],
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps(pod_spec)); pod_path = f.name

    print(f"Creating pod {pod_name}...")
    kubectl("apply", "-f", pod_path)

    print("Waiting for pod to start...")
    for _ in range(30):
        r = kubectl("get", "pod", pod_name, "-n", namespace,
                     "-o", "jsonpath={.status.phase}", capture=True)
        if r.stdout.strip() in ("Running", "Succeeded", "Failed"): break
        time.sleep(2)

    print("Streaming logs...\n")
    result = kubectl("logs", "-n", namespace, pod_name, "-f", capture=True)
    print(result.stdout)
    if result.stderr: print(result.stderr, file=sys.stderr)

    status = kubectl("get", "pod", pod_name, "-n", namespace,
                     "-o", "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}",
                     capture=True)
    exit_code = status.stdout.strip()

    print(f"\nCleaning up...")
    kubectl("delete", "pod", pod_name, "-n", namespace, "--ignore-not-found")
    kubectl("delete", "configmap", cm_name, "-n", namespace, "--ignore-not-found")

    if exit_code == "0":
        print(f"\n=== E2E {test_name.upper()} PASSED ===")
        return 0
    else:
        print(f"\n=== E2E {test_name.upper()} FAILED (exit code: {exit_code}) ===")
        return 1


def main():
    parser = argparse.ArgumentParser(description="End-to-end trainers test")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--test", default="smoke", choices=list(TESTS.keys()))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--accelerator", default="H200")
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--remote", default="baseten")
    parser.add_argument("--workspace-root", default=None)
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
    print(f"Test: {args.test}")

    sys.exit(run_test_pod(args.namespace, worker_url, job_id, args.test, args.model))


if __name__ == "__main__":
    main()
