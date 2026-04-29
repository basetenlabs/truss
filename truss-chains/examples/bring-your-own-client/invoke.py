"""Invoke the deployed Composable Chains BYO-Client demo.

Usage:
    BASETEN_API_KEY=... CHAIN_URL=https://chain-<id>.api.baseten.co/environments/production/run_remote \\
        python invoke.py "hello world"

The chain returns both the stub-path and the raw-httpx-path results plus the
descriptor metadata, so you can verify the bring-your-own-client pattern
matches the framework stub byte-for-byte.
"""

import json
import os
import sys

import requests


def main() -> None:
    api_key = os.environ.get("BASETEN_API_KEY")
    chain_url = os.environ.get("CHAIN_URL")
    if not api_key or not chain_url:
        sys.exit(
            "Set BASETEN_API_KEY and CHAIN_URL. CHAIN_URL is the chain's "
            "`<environment>/run_remote` endpoint shown after `chains push`."
        )

    text = sys.argv[1] if len(sys.argv) > 1 else "hello world"

    response = requests.post(
        chain_url,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={"text": text},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2))
    print()
    print("✓ MATCH" if result.get("match") else "✗ MISMATCH — bug in helpers!")


if __name__ == "__main__":
    main()
