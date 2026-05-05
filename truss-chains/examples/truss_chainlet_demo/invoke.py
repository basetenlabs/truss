"""Invoke the deployed Composable Chains TrussChainlet demo.

Usage:
    BASETEN_API_KEY=... CHAIN_URL=https://chain-<id>.api.baseten.co/.../run_remote \\
        python invoke.py "hello"
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

    text = sys.argv[1] if len(sys.argv) > 1 else "hello"
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

    expected_reversed = text[::-1]
    expected_upper = text.upper()
    ok = (
        result.get("via_chainletbase_reverser") == expected_reversed
        and result.get("via_truss_chainlet_echo") == expected_upper
    )
    if ok:
        print(
            f"✓ ChainletBase ({expected_reversed!r}) and TrussChainlet "
            f"({expected_upper!r}) both worked in one chain."
        )
    else:
        print(
            f"✗ Mismatch: expected reversed={expected_reversed!r}, "
            f"upper={expected_upper!r}."
        )


if __name__ == "__main__":
    main()
