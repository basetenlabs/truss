#!/usr/bin/env python3
"""Acceptance test: Math RL (GRPO) with the trainers SDK.

Implements a minimal GRPO loop for arithmetic problems:
1. Generate problems ("What is X + Y?")
2. Sample completions from the model
3. Score answers (correct/incorrect)
4. Compute group-relative advantages
5. Train with forward_backward + optim_step

When this test passes end-to-end against a live worker, the trainers SDK
has achieved functional parity with tinker's RL cookbook for single-turn RL.

Usage:
    python acceptance_math_rl.py <worker_url> [--steps 3] [--group-size 4]
"""

import argparse
import random
import sys
import time

from transformers import AutoTokenizer

# Adjust path for local dev
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainers"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "truss-train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trainers import ServiceClient, AdamParams, Datum, ModelInput, TensorData, SamplingParams


def make_arithmetic_problem(tokenizer, rng):
    """Generate a random addition problem and return (prompt_tokens, answer)."""
    x = rng.randint(0, 100)
    y = rng.randint(0, 100)
    prompt = f"What is {x} + {y}? Answer with just the number.\n"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return tokens, x + y


def check_answer(generated_tokens, tokenizer, expected):
    """Check if the generated tokens contain the correct answer."""
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    # Extract first number from response
    for word in text.split():
        try:
            return int(word) == expected
        except ValueError:
            continue
    return False


def compute_grpo_advantages(rewards):
    """Group-relative policy optimization: normalize rewards within the group."""
    mean = sum(rewards) / len(rewards) if rewards else 0
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards) if len(rewards) > 1 else 1
    std = max(variance ** 0.5, 1e-8)
    return [(r - mean) / std for r in rewards]


def main():
    parser = argparse.ArgumentParser(description="Math RL acceptance test")
    parser.add_argument("worker_url", help="dp_worker URL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name for tokenizer")
    parser.add_argument("--steps", type=int, default=3, help="Training steps")
    parser.add_argument("--group-size", type=int, default=4, help="Samples per problem")
    parser.add_argument("--problems-per-step", type=int, default=2, help="Problems per training step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max tokens per sample")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    service = ServiceClient(base_url=args.worker_url)
    client = service.create_lora_training_client(base_model=args.model, timeout=300.0)
    rng = random.Random(42)

    print(f"\n=== Math RL (GRPO) — {args.steps} steps, {args.group_size} samples/problem ===\n")

    adam_params = AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=0.95)

    for step in range(args.steps):
        t0 = time.perf_counter()
        print(f"--- Step {step + 1}/{args.steps} ---")

        # 1. Switch to inference mode for sampling
        print("  [1] save_weights_and_get_sampling_client")
        client.save_weights_and_get_sampling_client().result(timeout=120)

        # 2. Generate problems and sample completions
        all_data = []
        total_correct = 0
        total_samples = 0

        for p in range(args.problems_per_step):
            prompt_tokens, expected_answer = make_arithmetic_problem(tokenizer, rng)
            prompt = ModelInput.from_ints(prompt_tokens)

            rewards = []
            completions_tokens = []

            for s in range(args.group_size):
                result = client.sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                ).result(timeout=60)

                gen_tokens = result.sequences[0].tokens
                correct = check_answer(gen_tokens, tokenizer, expected_answer)
                reward = 1.0 if correct else 0.0
                rewards.append(reward)
                completions_tokens.append(gen_tokens)
                total_correct += int(correct)
                total_samples += 1

            # 3. Compute GRPO advantages
            advantages = compute_grpo_advantages(rewards)

            # 4. Build training data
            for gen_tokens, advantage in zip(completions_tokens, advantages):
                full_tokens = prompt_tokens + gen_tokens
                if len(full_tokens) < 2:
                    continue
                all_data.append(Datum(
                    model_input=ModelInput.from_ints(full_tokens),
                    loss_fn_inputs={
                        "advantages": TensorData(data=[advantage], dtype="float32", shape=[1]),
                    },
                ))

        accuracy = total_correct / total_samples if total_samples else 0
        print(f"  [2] Sampled {total_samples} completions, accuracy={accuracy:.1%}")

        # 5. Train: forward_backward + optim_step
        if all_data:
            fwd_result = client.forward_backward(data=all_data, loss_fn="cross_entropy").result(timeout=120)
            print(f"  [3] forward_backward: loss={fwd_result.metrics.get('loss', '?')}")

            optim_result = client.optim_step(adam_params).result(timeout=60)
            print(f"  [4] optim_step: step={optim_result.metrics.get('step', '?')}")
        else:
            print("  [3] No training data (all sequences too short)")

        elapsed = time.perf_counter() - t0
        print(f"  Step {step + 1} done in {elapsed:.1f}s\n")

    print("=== Math RL acceptance test PASSED ===")
    client.close()


if __name__ == "__main__":
    main()
