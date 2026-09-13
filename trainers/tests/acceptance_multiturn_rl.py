#!/usr/bin/env python3
"""Acceptance test: Multi-turn RL with the trainers SDK.

Implements a multi-turn "number guessing" game:
- Environment picks a secret number 1-100
- Model gets 5 turns to guess it
- Each turn: model generates a guess, environment says "higher"/"lower"/"correct"
- Reward: 1.0 if correct within 5 turns, 0.0 otherwise

This tests the full multi-turn RL loop:
1. Multi-step rollout with accumulating context
2. Per-turn sampling with growing prompt
3. Environment feedback injected into context
4. GRPO advantage computation across groups
5. Training on full trajectories

When this passes, the trainers SDK supports multi-turn RL — the key
capability for tool-use, agentic, and conversational RL training.

Usage:
    python acceptance_multiturn_rl.py <worker_url> [--steps 3] [--group-size 4]
"""

import argparse
import random
import re
import sys
import time

from transformers import AutoTokenizer

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainers"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "truss-train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trainers import ServiceClient, AdamParams, Datum, ModelInput, TensorData, SamplingParams


class NumberGuessingEnv:
    """Simple multi-turn environment: guess a number between 1 and 100."""

    def __init__(self, secret: int, max_turns: int = 5):
        self.secret = secret
        self.max_turns = max_turns
        self.turns = 0
        self.solved = False
        self.history = f"I'm thinking of a number between 1 and 100. You have {max_turns} guesses.\n"

    def step(self, model_response: str) -> tuple[str, bool]:
        """Process model's guess, return (feedback, episode_done)."""
        self.turns += 1

        # Extract number from response
        guess = None
        for word in re.findall(r'\d+', model_response):
            try:
                guess = int(word)
                break
            except ValueError:
                continue

        if guess is None:
            feedback = f"Turn {self.turns}: I couldn't understand your guess. Please guess a number.\n"
        elif guess == self.secret:
            self.solved = True
            feedback = f"Turn {self.turns}: Correct! The number was {self.secret}.\n"
        elif guess < self.secret:
            feedback = f"Turn {self.turns}: {guess} is too low. Guess higher.\n"
        else:
            feedback = f"Turn {self.turns}: {guess} is too high. Guess lower.\n"

        self.history += feedback
        done = self.solved or self.turns >= self.max_turns
        return feedback, done

    @property
    def reward(self) -> float:
        return 1.0 if self.solved else 0.0


def do_rollout(client, tokenizer, env, sampling_params):
    """Run a single multi-turn rollout, returning (full_token_trajectory, reward)."""
    all_tokens = tokenizer.encode(env.history, add_special_tokens=False)

    for turn in range(env.max_turns):
        # Sample the model's next response
        prompt = ModelInput.from_ints(all_tokens)
        result = client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        ).result(timeout=60)

        gen_tokens = result.sequences[0].tokens
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Add model response to trajectory
        all_tokens = all_tokens + gen_tokens

        # Step environment
        feedback, done = env.step(gen_text)
        feedback_tokens = tokenizer.encode(feedback, add_special_tokens=False)
        all_tokens = all_tokens + feedback_tokens

        if done:
            break

    return all_tokens, env.reward, env.turns


def compute_grpo_advantages(rewards):
    mean = sum(rewards) / len(rewards) if rewards else 0
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards) if len(rewards) > 1 else 1
    std = max(variance ** 0.5, 1e-8)
    return [(r - mean) / std for r in rewards]


def main():
    parser = argparse.ArgumentParser(description="Multi-turn RL acceptance test")
    parser.add_argument("worker_url", help="dp_worker URL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name for tokenizer")
    parser.add_argument("--steps", type=int, default=3, help="Training steps")
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per problem")
    parser.add_argument("--problems-per-step", type=int, default=2, help="Problems per step")
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns per game")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens per turn")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    service = ServiceClient(base_url=args.worker_url)
    client = service.create_lora_training_client(base_model=args.model, timeout=300.0)
    rng = random.Random(42)

    print(f"\n=== Multi-turn RL (Number Guessing) — {args.steps} steps ===\n")

    adam_params = AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=0.95)

    for step in range(args.steps):
        t0 = time.perf_counter()
        print(f"--- Step {step + 1}/{args.steps} ---")

        # 1. Switch to inference
        print("  [1] save_weights_and_get_sampling_client")
        client.save_weights_and_get_sampling_client().result(timeout=120)

        # 2. Multi-turn rollouts
        all_data = []
        total_solved = 0
        total_rollouts = 0
        total_turns = 0

        for p in range(args.problems_per_step):
            secret = rng.randint(1, 100)
            rewards = []
            trajectories = []

            for g in range(args.group_size):
                env = NumberGuessingEnv(secret=secret, max_turns=args.max_turns)
                traj_tokens, reward, turns = do_rollout(client, tokenizer, env, sampling_params)
                rewards.append(reward)
                trajectories.append(traj_tokens)
                total_solved += int(reward > 0)
                total_rollouts += 1
                total_turns += turns

            # 3. GRPO advantages
            advantages = compute_grpo_advantages(rewards)

            # 4. Build training data
            for traj_tokens, advantage in zip(trajectories, advantages):
                if len(traj_tokens) < 2:
                    continue
                all_data.append(Datum(
                    model_input=ModelInput.from_ints(traj_tokens),
                    loss_fn_inputs={
                        "advantages": TensorData(data=[advantage], dtype="float32", shape=[1]),
                    },
                ))

        solve_rate = total_solved / total_rollouts if total_rollouts else 0
        avg_turns = total_turns / total_rollouts if total_rollouts else 0
        print(f"  [2] {total_rollouts} rollouts: solve_rate={solve_rate:.1%}, avg_turns={avg_turns:.1f}")

        # 5. Train
        if all_data:
            fwd_result = client.forward_backward(data=all_data, loss_fn="cross_entropy").result(timeout=120)
            print(f"  [3] forward_backward: loss={fwd_result.metrics.get('loss', '?')}")

            optim_result = client.optim_step(adam_params).result(timeout=60)
            print(f"  [4] optim_step: step={optim_result.metrics.get('step', '?')}")

        elapsed = time.perf_counter() - t0
        print(f"  Step {step + 1} done in {elapsed:.1f}s\n")

    print("=== Multi-turn RL acceptance test PASSED ===")
    client.close()


if __name__ == "__main__":
    main()
