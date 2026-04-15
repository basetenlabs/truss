#!/usr/bin/env python3
"""Example: create a sampling client and get a prediction.

Usage:
    TRAINERS_BASE_URL=http://<worker-url> python sample_prediction.py
    python sample_prediction.py --base-url http://<worker-url> --model Qwen/Qwen3-8B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainers"))

import trainers
from trainers import EncodedTextChunk, ModelInput, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Sample a prediction from a deployed model")
    parser.add_argument("--base-url", help="Worker base URL (or set TRAINERS_BASE_URL)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Create a sampling client (no training client needed)
    service_client = trainers.ServiceClient(base_url=args.base_url)
    sampling_client = service_client.create_sampling_client(base_model=args.model)

    # Tokenize the prompt
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = sampling_client.get_tokenizer()
    tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f'Prompt: "{args.prompt}"')

    # Get a prediction
    print("Sampling...")
    response = sampling_client.sample(
        prompt=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    # Decode and print
    print()
    for seq in response.sequences:
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        print(text)


if __name__ == "__main__":
    main()
