"""Run a small built-in scenario matrix against the integration Truss servers."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass

import requests
from baseten_performance_client import (
    EndpointPool,
    HttpClientWrapper,
    PerformanceClient,
    RequestProcessingPreference,
)

DEFAULT_ENDPOINT_URLS = [
    "https://model-3ydyzel3.api.baseten.co/deployment/wpjek0l/sync",
    "https://model-3ydyzel3.api.baseten.co/deployment/wnpx071/sync",
    "https://model-3ydyzel3.api.baseten.co/deployment/wom96gn/sync",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    texts_per_operation: int
    batch_size: int
    max_concurrent_requests: int
    pin_initial_endpoint_once: bool


SCENARIOS = [
    Scenario(
        name="pinned_batch_size_1",
        texts_per_operation=9,
        batch_size=1,
        max_concurrent_requests=9,
        pin_initial_endpoint_once=True,
    ),
    Scenario(
        name="pinned_batch_size_3",
        texts_per_operation=12,
        batch_size=3,
        max_concurrent_requests=6,
        pin_initial_endpoint_once=True,
    ),
    Scenario(
        name="unpinned_batch_size_1",
        texts_per_operation=9,
        batch_size=1,
        max_concurrent_requests=9,
        pin_initial_endpoint_once=False,
    ),
    Scenario(
        name="unpinned_batch_size_3",
        texts_per_operation=12,
        batch_size=3,
        max_concurrent_requests=6,
        pin_initial_endpoint_once=False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a built-in embedding failover scenario matrix against the "
            "three Baseten integration Truss servers."
        )
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BASETEN_API_KEY"),
        help="Baseten API key. Defaults to BASETEN_API_KEY.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=120.0,
        help="Duration per scenario in seconds.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Sleep interval between top-level operations.",
    )
    parser.add_argument(
        "--http-version",
        type=int,
        default=1,
        choices=(1, 2),
        help="HTTP version for request and health-check clients.",
    )
    return parser.parse_args()


def build_client(api_key: str, http_version: int) -> PerformanceClient:
    request_wrapper = HttpClientWrapper(http_version=http_version)
    health_wrapper = HttpClientWrapper(http_version=http_version)

    endpoint_pool = EndpointPool(
        endpoint_urls=DEFAULT_ENDPOINT_URLS,
        client_wrapper=health_wrapper,
        endpoint_weights=[1.0, 1.0, 1.0],
        deployment_health_path="/health",
        health_check_interval_s=2.0,
        health_check_timeout_s=0.75,
        health_check_retries=2,
        health_fail_on_first=False,
        deployment_timeout_is_no_vote=True,
    )

    return PerformanceClient(
        base_url=DEFAULT_ENDPOINT_URLS[0],
        api_key=api_key,
        http_version=http_version,
        client_wrapper=request_wrapper,
        endpoint_pool=endpoint_pool,
    )


def build_preference(scenario: Scenario) -> RequestProcessingPreference:
    return RequestProcessingPreference(
        max_concurrent_requests=scenario.max_concurrent_requests,
        batch_size=scenario.batch_size,
        pin_initial_endpoint_once=scenario.pin_initial_endpoint_once,
        timeout_s=12.0,
        hedge_budget_pct=0.0,
        retry_budget_pct=1.0,
        max_retries=3,
        initial_backoff_ms=125,
    )


def extract_server_names(response_headers: list[dict[str, str]]) -> list[str]:
    return [
        header_map.get("x-integration-server-name", "unknown")
        for header_map in response_headers
    ]


def run_scenario(
    client: PerformanceClient,
    scenario: Scenario,
    *,
    duration_s: float,
    interval_s: float,
) -> int:
    preference = build_preference(scenario)
    deadline = time.time() + duration_s

    successes = 0
    failures = 0
    chosen_server_counter: Counter[str] = Counter()
    total_times: list[float] = []

    print()
    print(
        f"Scenario {scenario.name}: texts_per_operation={scenario.texts_per_operation} "
        f"batch_size={scenario.batch_size} "
        f"max_concurrent_requests={scenario.max_concurrent_requests} "
        f"pin_initial_endpoint_once={scenario.pin_initial_endpoint_once}"
    )

    operation_index = 0
    while time.time() < deadline:
        inputs = [
            (
                f"integration-failover-probe scenario={scenario.name} "
                f"operation={operation_index} item={item_index}"
            )
            for item_index in range(scenario.texts_per_operation)
        ]

        start = time.time()
        try:
            response = client.embed(input=inputs, model="model", preference=preference)
            elapsed = time.time() - start
            total_times.append(elapsed)
            successes += 1

            server_names = extract_server_names(response.response_headers)
            distinct_server_names = sorted(set(server_names))
            for server_name in distinct_server_names:
                chosen_server_counter[server_name] += 1

            batch_times = response.individual_request_times or []
            batch_time_summary = (
                f"min={min(batch_times):.3f}s median={statistics.median(batch_times):.3f}s "
                f"max={max(batch_times):.3f}s"
                if batch_times
                else "no batch timings"
            )
            print(
                f"[ok] scenario={scenario.name} op={operation_index:04d} total={elapsed:.3f}s "
                f"servers={distinct_server_names} {batch_time_summary}"
            )
        except requests.exceptions.HTTPError as exc:
            failures += 1
            elapsed = time.time() - start
            print(
                f"[http-error] scenario={scenario.name} op={operation_index:04d} "
                f"total={elapsed:.3f}s error={exc}"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            elapsed = time.time() - start
            print(
                f"[error] scenario={scenario.name} op={operation_index:04d} "
                f"total={elapsed:.3f}s error={exc}"
            )

        operation_index += 1
        time.sleep(interval_s)

    print(
        f"Scenario {scenario.name} summary: successes={successes} failures={failures}"
    )
    if total_times:
        print(
            f"  total_time_s min={min(total_times):.3f} "
            f"median={statistics.median(total_times):.3f} "
            f"max={max(total_times):.3f}"
        )
    if chosen_server_counter:
        print("  chosen servers:")
        for server_name, count in chosen_server_counter.most_common():
            print(f"    {server_name}: {count}")

    return failures


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set BASETEN_API_KEY.")

    client = build_client(args.api_key, args.http_version)

    print("Testing EndpointPool failover with endpoints:")
    for endpoint_url in DEFAULT_ENDPOINT_URLS:
        print(f"  - {endpoint_url}")
    print(
        "Shared client settings: "
        "health_check_interval_s=2.0 "
        "health_check_timeout_s=0.75 "
        "health_check_retries=2 "
        "endpoint_weights=[1.0, 1.0, 1.0]"
    )

    total_failures = 0
    for scenario in SCENARIOS:
        total_failures += run_scenario(
            client, scenario, duration_s=args.duration_s, interval_s=args.interval_s
        )

    print()
    print(f"Finished scenario matrix. total_failures={total_failures}")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
